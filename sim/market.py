"""The daily loop. Days run in synchronized rounds:

1. Inbox review: everyone reviews incoming likes (best-first or newest-first) up to their
   attention budget, and likes back according to their pre-drawn decision. A like-back is a
   match, and liking back is free (it doesn't count against the daily like cap).
2. Browse: everyone gets a ranked list and swipes until the list ends, they hit the like cap,
   or they lose patience. Likes sent today arrive in the recipient's inbox tomorrow.
3. End of day: unreviewed likes vanish (classroom) or carry over (Hinge mode).

A final review-only round after the last day processes the last day's likes.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rankers.base import MarketState, Ranker, Swipes
from sim.preferences import OTHER, SIDE_ID, SIDES, World, rng_for


@dataclass
class Inbox:
    """Pending likes for one side, as parallel arrays (one entry per like)."""

    recipient: np.ndarray
    sender: np.ndarray
    day_sent: np.ndarray

    @classmethod
    def empty(cls) -> "Inbox":
        return cls(*(np.zeros(0, int) for _ in range(3)))

    def __len__(self) -> int:
        return len(self.recipient)

    def where(self, mask: np.ndarray) -> "Inbox":
        return Inbox(self.recipient[mask], self.sender[mask], self.day_sent[mask])

    def add(self, recipient: np.ndarray, sender: np.ndarray, day: int) -> "Inbox":
        return Inbox(
            np.concatenate([self.recipient, recipient]),
            np.concatenate([self.sender, sender]),
            np.concatenate([self.day_sent, np.full(len(recipient), day)]),
        )


class Log:
    """Every event, accumulated as array chunks and joined at the end."""

    BROWSE = ("day", "side", "viewer", "shown", "position", "liked")
    INBOX = ("side", "recipient", "sender", "day_sent", "day_reviewed", "liked_back")

    def __init__(self) -> None:
        self.browse = {k: [] for k in self.BROWSE}
        self.inbox = {k: [] for k in self.INBOX}

    def add_browse(self, day, side, viewer, shown, position, liked) -> None:
        for k, v in zip(
            self.BROWSE,
            (np.full(len(viewer), day), np.full(len(viewer), side), viewer, shown, position, liked),
        ):
            self.browse[k].append(v)

    def add_inbox(self, side: int, box: Inbox, day_reviewed: int, liked_back) -> None:
        n = len(box)
        values = (
            np.full(n, side),
            box.recipient,
            box.sender,
            box.day_sent,
            np.full(n, day_reviewed),
            np.broadcast_to(liked_back, n),
        )
        for k, v in zip(self.INBOX, values):
            self.inbox[k].append(v)

    @staticmethod
    def _join(chunks: dict) -> dict[str, np.ndarray]:
        return {k: np.concatenate(v) if v else np.zeros(0, int) for k, v in chunks.items()}


@dataclass
class SimResult:
    ranker: str
    days: int
    browse: dict[str, np.ndarray]  # one row per profile viewed while browsing
    inbox: dict[str, np.ndarray]  # one row per like sent; day_reviewed = -1 if never reviewed
    matched: np.ndarray  # (n_a, n_b) bool
    exposure: dict[str, np.ndarray]  # per user: share of the other side they've seen

    def to_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Event logs as DataFrames (the training data for Phases 3-4)."""
        browse = pd.DataFrame(self.browse)
        a = np.where(browse.side == 0, browse.viewer, browse.shown)
        b = np.where(browse.side == 0, browse.shown, browse.viewer)
        browse["matched"] = self.matched[a, b]
        browse["ranker"] = self.ranker
        inbox = pd.DataFrame(self.inbox)
        inbox["liked_back"] = inbox.liked_back.astype(bool)
        return browse, inbox


class Market:
    def __init__(self, world: World, ranker: Ranker) -> None:
        self.world, self.cfg, self.ranker = world, world.cfg, ranker
        n = self.cfg.n_per_side
        self.people = {"a": world.a, "b": world.b}
        self.decides = {"a": world.DA, "b": world.DB}
        self.seen = {s: np.zeros((n, n), bool) for s in SIDES}  # [viewer, other side]
        self.matched = np.zeros((n, n), bool)  # [a, b]
        self.likes_received = {s: np.zeros(n, int) for s in SIDES}
        self.inbox = {s: Inbox.empty() for s in SIDES}
        self.arriving: dict[str, tuple] = {}  # today's likes, delivered to inboxes tonight
        self.rng = rng_for(world.seed, "ranker")
        # Patience gets its own stream with a fixed number of draws per day, so every ranker
        # faces the same patience for the same person on the same day.
        self.patience_rng = rng_for(world.seed, "patience")
        self.log = Log()
        ranker.reset(world, self.rng)

    def matched_from(self, s: str) -> np.ndarray:
        """Match matrix oriented [person on side s, person on the other side]."""
        return self.matched if s == "a" else self.matched.T

    def run(self) -> SimResult:
        for day in range(self.cfg.days):
            swipes = self.review(day) + self.browse(day)
            self.ranker.observe(_join_swipes(swipes))
            self.end_of_day(drop=not self.cfg.carry_over)
        self.review(self.cfg.days)
        self.end_of_day(drop=True)
        return SimResult(
            ranker=self.ranker.name,
            days=self.cfg.days,
            browse=Log._join(self.log.browse),
            inbox=Log._join(self.log.inbox),
            matched=self.matched,
            exposure={s: self.seen[s].mean(axis=1) for s in SIDES},
        )

    def review(self, day: int) -> list[Swipes]:
        swipes = []
        for s in SIDES:
            box = self.inbox[s]
            # Likes from someone you've since matched with are resolved, not waiting.
            stale = self.matched_from(s)[box.recipient, box.sender]
            self.log.add_inbox(SIDE_ID[s], box.where(stale), day_reviewed=-1, liked_back=False)
            box = box.where(~stale)
            if not len(box):
                self.inbox[s] = box
                continue

            if self.cfg.inbox_order == "best":
                priority = -self.world.probs(s)[box.recipient, box.sender]
            else:
                priority = -box.day_sent
            # Sort by recipient, then priority (ties broken at random); then each recipient
            # reviews the first `attention` likes in their group.
            order = np.lexsort((self.rng.random(len(box)), priority, box.recipient))
            rec = box.recipient[order]
            rank_in_group = np.arange(len(rec)) - np.searchsorted(rec, rec)
            take = np.zeros(len(box), bool)
            take[order] = rank_in_group < self.people[s].attention[rec]

            done = box.where(take)
            liked_back = self.decides[s][done.recipient, done.sender]
            self.seen[s][done.recipient, done.sender] = True
            self.matched_from(s)[done.recipient[liked_back], done.sender[liked_back]] = True
            self.log.add_inbox(SIDE_ID[s], done, day_reviewed=day, liked_back=liked_back)
            swipes.append(
                Swipes(np.full(len(done), SIDE_ID[s]), done.recipient, done.sender, liked_back)
            )
            self.inbox[s] = box.where(~take)
        return swipes

    def browse(self, day: int) -> list[Swipes]:
        eligible = {}
        for s in SIDES:
            eligible[s] = ~self.seen[s]
            eligible[s][self.inbox[s].recipient, self.inbox[s].sender] = False
        state = MarketState(self.world, day, self.likes_received, eligible)
        swipes, new_likes = [], {}
        for s in SIDES:
            n, L = self.people[s].n, self.cfg.list_len
            lists = self.ranker.lists(s, state, eligible[s], L)

            valid = lists >= 0
            rows = np.arange(n)[:, None]
            would_like = self.decides[s][rows, np.where(valid, lists, 0)] & valid
            # Stop right after the like that hits the cap, or when patience runs out.
            likes_so_far = np.cumsum(would_like, axis=1)
            hit_cap = likes_so_far >= self.cfg.like_cap
            cap_stop = np.where(hit_cap.any(axis=1), hit_cap.argmax(axis=1) + 1, L)
            if self.cfg.quit_prob > 0:
                patience = self.patience_rng.geometric(self.cfg.quit_prob, n)
            else:
                patience = np.full(n, L)
            n_viewed = np.minimum.reduce([cap_stop, patience, valid.sum(axis=1)])
            viewed = np.arange(L)[None, :] < n_viewed[:, None]

            viewer, position = np.nonzero(viewed)
            shown = lists[viewer, position]
            liked = would_like[viewer, position]
            self.seen[s][viewer, shown] = True
            self.log.add_browse(day, SIDE_ID[s], viewer, shown, position, liked)
            swipes.append(Swipes(np.full(len(viewer), SIDE_ID[s]), viewer, shown, liked))
            new_likes[s] = (viewer[liked], shown[liked])

        # Apply after both sides browse, so both sides' rankers saw the same morning state.
        for s, (sender, recipient) in new_likes.items():
            o = OTHER[s]
            self.likes_received[o] += np.bincount(recipient, minlength=self.people[o].n)
            self.arriving[o] = (recipient, sender, day)
        return swipes

    def end_of_day(self, drop: bool) -> None:
        for s in SIDES:
            if drop:
                self.log.add_inbox(SIDE_ID[s], self.inbox[s], day_reviewed=-1, liked_back=False)
                self.inbox[s] = Inbox.empty()
            if s in self.arriving:
                self.inbox[s] = self.inbox[s].add(*self.arriving.pop(s))


def _join_swipes(parts: list[Swipes]) -> Swipes:
    return Swipes(
        *(
            np.concatenate([getattr(p, f) for p in parts])
            for f in ("side", "swiper", "target", "liked")
        )
    )


def simulate(world: World, ranker: Ranker) -> SimResult:
    return Market(world, ranker).run()
