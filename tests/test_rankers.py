import numpy as np
import pandas as pd
import pytest

from rankers.base import MarketState, Ranker, Swipes
from rankers.baselines import OneSidedOracle, PopularityRanker, RandomRanker, ReciprocalOracle
from rankers.elo import EloRanker
from sim.config import Config
from sim.market import simulate
from sim.metrics import summarize
from sim.preferences import build_world

CFG = Config(n_per_side=300, days=10)


def ready(ranker, world, seed=0):
    ranker.reset(world, np.random.default_rng(seed))
    return ranker


def state(world, likes_received=None):
    n = world.cfg.n_per_side
    zeros = {"a": np.zeros(n, int), "b": np.zeros(n, int)}
    return MarketState(world, day=0, likes_received=likes_received or zeros)


@pytest.fixture(scope="module")
def world():
    return build_world(CFG, seed=0)


def test_elo_update_by_hand(world):
    elo = ready(EloRanker(), world)
    elo.rating["a"][0] = 1600  # swiper on side A
    elo.rating["b"][[0, 1]] = 1400  # two targets on side B
    elo.observe(Swipes(np.array([0, 0]), np.array([0, 0]), np.array([0, 1]), np.array([1, 0])))
    expected = 1 / (1 + 10 ** ((1600 - 1400) / 400))  # 0.240: B "should" lose to a 1600
    assert elo.rating["b"][0] == pytest.approx(1400 + 32 * (1 - expected))  # liked: +24.3
    assert elo.rating["b"][1] == pytest.approx(1400 + 32 * (0 - expected))  # passed: -7.7
    assert elo.rating["a"][0] == 1600  # only the person being swiped on moves


def test_elo_applies_a_days_swipes_from_morning_ratings(world):
    elo = ready(EloRanker(), world)
    # Two side-B swipers both like side-A person 5 on the same day: both use the 1500 rating.
    elo.observe(Swipes(np.array([1, 1]), np.array([0, 1]), np.array([5, 5]), np.array([1, 1])))
    assert elo.rating["a"][5] == pytest.approx(1500 + 2 * 32 * 0.5)


def test_elo_shows_people_at_a_similar_percentile(world):
    elo = ready(EloRanker(), world)
    n = CFG.n_per_side
    elo.rating["a"] = np.arange(n, dtype=float)  # viewer i is at percentile ~ i / n
    elo.rating["b"] = np.arange(n, dtype=float)[::-1].copy()  # candidate j at ~ (n - 1 - j) / n
    lists = elo.lists("a", state(world), np.ones((n, n), bool), 10)
    # The 9 ranked slots for viewer 0 (bottom of A) should be the bottom of B (high j).
    assert set(lists[0, :9]) == set(range(n - 9, n))


def test_popularity_breaks_ties_differently_per_viewer(world):
    pop = ready(PopularityRanker(), world)
    n = CFG.n_per_side
    lists = pop.lists("a", state(world), np.ones((n, n), bool), 10)
    assert len({tuple(row) for row in lists[:, :9]}) > n // 2


def test_popularity_shows_most_liked_first(world):
    pop = ready(PopularityRanker(), world)
    n = CFG.n_per_side
    received = {"a": np.zeros(n, int), "b": np.arange(n)}  # B person j has j likes
    lists = pop.lists("a", state(world, received), np.ones((n, n), bool), 10)
    np.testing.assert_array_equal(lists[:, 0], n - 1)
    assert set(lists[3, :9]) == set(range(n - 9, n))


def test_exploration_fills_every_tenth_slot_at_random(world):
    pop = ready(PopularityRanker(), world)
    n = CFG.n_per_side
    received = {"a": np.zeros(n, int), "b": np.arange(n)}
    lists = pop.lists("a", state(world, received), np.ones((n, n), bool), 10)
    explored = lists[:, 9]
    assert (explored < n - 9).all()  # never one of the top 9
    assert len(set(explored)) > n // 2  # and different for different viewers


def test_oracles_rank_by_true_probabilities(world):
    n = CFG.n_per_side
    eligible = np.ones((n, n), bool)
    one = ready(OneSidedOracle(), world).lists("b", state(world), eligible, 5)
    np.testing.assert_array_equal(one, np.argsort(-world.Q, axis=1)[:, :5])
    rec = ready(ReciprocalOracle(), world).lists("b", state(world), eligible, 5)
    np.testing.assert_array_equal(rec, np.argsort(-(world.Q * world.P.T), axis=1)[:, :5])


def test_lists_respect_eligibility(world):
    n = CFG.n_per_side
    eligible = np.zeros((n, n), bool)
    eligible[:, :3] = True  # only three candidates left for anyone
    lists = ready(RandomRanker(), world).lists("a", state(world), eligible, 10)
    assert set(np.unique(lists[:, :3])) <= {0, 1, 2}
    assert (lists[:, 3:] == -1).all()


@pytest.fixture(scope="module")
def market_results():
    rankers = [RandomRanker, PopularityRanker, EloRanker, OneSidedOracle, ReciprocalOracle]
    rows = []
    for seed in range(3):
        w = build_world(CFG, seed)
        for make in rankers:
            ranker: Ranker = make()
            rows.append({"ranker": ranker.name, **summarize(w, simulate(w, ranker))})
    return pd.DataFrame(rows).groupby("ranker").matches.mean()


@pytest.mark.parametrize("other", ["random", "popularity", "elo"])
def test_reciprocal_oracle_beats_baselines(market_results, other):
    assert market_results["reciprocal_oracle"] >= market_results[other]


def test_reciprocal_beats_one_sided_oracle(market_results):
    assert market_results["reciprocal_oracle"] > market_results["one_sided_oracle"]
