"""Reference rankers."""

from rankers.base import MarketState, Ranker


class RandomRanker(Ranker):
    """The floor: a fresh random order every day."""

    name = "random"

    def scores(self, side: str, state: MarketState):
        n = state.world.cfg.n_per_side
        return self.rng.random((n, n))
