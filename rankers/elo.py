"""Elo, the way Tinder reportedly used it: each swipe is a "game" the person being swiped on
wins (a like) or loses (a pass), against the swiper's rating."""

import numpy as np

from rankers.base import MarketState, Ranker, Swipes
from sim.preferences import OTHER, SIDES


def _percentiles(rating: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rank within one's own side, in (0, 1), with random tie-breaking."""
    order = np.lexsort((rng.random(len(rating)), rating))
    pct = np.empty(len(rating))
    pct[order] = (np.arange(len(rating)) + 0.5) / len(rating)
    return pct


class EloRanker(Ranker):
    name = "elo"
    explore = True

    def reset(self, world, rng) -> None:
        super().reset(world, rng)
        n, start = world.cfg.n_per_side, world.cfg.elo_start
        self.rating = {s: np.full(n, start) for s in SIDES}

    def observe(self, swipes: Swipes) -> None:
        """Only the person swiped on moves. All of a day's swipes use the morning ratings."""
        k = self.world.cfg.elo_k
        delta = {s: np.zeros_like(self.rating[s]) for s in SIDES}
        for i, s in enumerate(SIDES):  # s = the swiper's side
            o, mine = OTHER[s], swipes.side == i
            swiper = self.rating[s][swipes.swiper[mine]]
            target = self.rating[o][swipes.target[mine]]
            expected = 1 / (1 + 10 ** ((swiper - target) / 400))  # target's chance to "win"
            np.add.at(delta[o], swipes.target[mine], k * (swipes.liked[mine] - expected))
        for s in SIDES:
            self.rating[s] += delta[s]

    def scores(self, side: str, state: MarketState) -> np.ndarray:
        # Ratings aren't comparable across sides (side B is pickier, so side A's ratings
        # sink), so match on percentile within each side instead of raw rating.
        me = _percentiles(self.rating[side], self.rng)
        them = _percentiles(self.rating[OTHER[side]], self.rng)
        return -np.abs(me[:, None] - them[None, :])
