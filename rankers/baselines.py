"""Reference rankers: the floor (random), a naive heuristic (popularity), and two oracles that
read the true preferences. The oracles are reference points, not something an app could run."""

import numpy as np

from rankers.base import MarketState, Ranker
from sim.preferences import OTHER


class RandomRanker(Ranker):
    """The floor: a fresh random order every day."""

    name = "random"

    def scores(self, side: str, state: MarketState) -> np.ndarray:
        n = state.world.cfg.n_per_side
        return self.rng.random((n, n))


class PopularityRanker(Ranker):
    """Most likes received first. Counts are integers, so adding U[0, 1) only breaks ties,
    and a fresh draw per viewer means day 1 (everyone at zero) isn't the same list for all."""

    name = "popularity"
    explore = True

    def scores(self, side: str, state: MarketState) -> np.ndarray:
        received = state.likes_received[OTHER[side]]
        return received[None, :] + self.rng.random((len(received), len(received)))


class OneSidedOracle(Ranker):
    """Rank by how much I'd like them, ignoring whether they'd like me back."""

    name = "one_sided_oracle"

    def scores(self, side: str, state: MarketState) -> np.ndarray:
        return state.world.probs(side)


class ReciprocalOracle(Ranker):
    """Rank by the true chance of a match: P(I like them) x P(they like me back).
    The best *pairwise* ranker, but not the best possible: it ignores congestion, so it will
    happily send everyone to the same people."""

    name = "reciprocal_oracle"

    def reset(self, world, rng) -> None:
        super().reset(world, rng)
        self.match_prob = {"a": world.P * world.Q.T, "b": world.Q * world.P.T}

    def scores(self, side: str, state: MarketState) -> np.ndarray:
        return self.match_prob[side]
