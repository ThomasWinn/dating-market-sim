"""What every ranker shares: turning a score matrix into each viewer's ordered list."""

from dataclasses import dataclass

import numpy as np

from sim.preferences import World


@dataclass
class MarketState:
    """What a ranker may look at when building today's lists."""

    world: World  # the oracles read true P and Q from here; other rankers must not
    day: int
    likes_received: dict[str, np.ndarray]  # browse likes received so far, per side


@dataclass
class Swipes:
    """One day of decisions, from browsing and from inbox review. Elo learns from these."""

    side: np.ndarray  # 0 if the swiper is on side A, 1 if on side B
    swiper: np.ndarray
    target: np.ndarray
    liked: np.ndarray


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Column indices of each row's k highest scores, best first."""
    k = min(k, scores.shape[1])
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, idx, axis=1), axis=1, kind="stable")
    return np.take_along_axis(idx, order, axis=1)


def exploration_slots(k: int, frac: float) -> list[int]:
    """Spread random slots through the list (every 10th at 10%), so scrollers reach them."""
    step = round(1 / frac)
    return list(range(step - 1, k, step))


class Ranker:
    name = "ranker"
    explore = False  # fill some slots at random so low scorers aren't starved forever

    def reset(self, world: World, rng: np.random.Generator) -> None:
        self.world, self.rng = world, rng

    def scores(self, side: str, state: MarketState) -> np.ndarray:
        """(n_viewers, n_candidates): higher = show earlier."""
        raise NotImplementedError

    def observe(self, swipes: Swipes) -> None:
        pass

    def lists(self, side: str, state: MarketState, eligible: np.ndarray, k: int) -> np.ndarray:
        """(n_viewers, k) candidate indices in display order; -1 pads when nobody is left."""
        s = np.where(eligible, self.scores(side, state), -np.inf)
        slots = exploration_slots(k, self.world.cfg.explore_frac) if self.explore else []
        out = np.full((len(s), k), -1)

        main = top_k(s, k - len(slots))
        ok = np.take_along_axis(s, main, axis=1) > -np.inf
        out[:, [p for p in range(k) if p not in slots]] = np.where(ok, main, -1)

        if slots:
            r = np.where(eligible, self.rng.random(s.shape), -np.inf)
            np.put_along_axis(r, main, -np.inf, axis=1)  # don't pick someone twice
            extra = top_k(r, len(slots))
            ok = np.take_along_axis(r, extra, axis=1) > -np.inf
            out[:, slots] = np.where(ok, extra, -1)

        # Push any -1 gaps to the end so everyone's real candidates come first.
        return np.take_along_axis(out, np.argsort(out < 0, axis=1, kind="stable"), axis=1)
