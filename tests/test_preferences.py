import numpy as np
import pytest

from sim.config import IDEAL_POINT, MORE_IS_BETTER, Config, with_w
from sim.metrics import gini
from sim.population import Side
from sim.preferences import build_world, like_probs, quantile_thresholds, scores, taste

N = 400
CFG = Config(n_per_side=N)


def test_taste_by_hand():
    def person(traits, ideals, importance):
        one = np.ones(1)
        return Side(
            one, one, np.array([traits]), np.array([ideals]), np.array([importance]), one, one
        )

    viewer = person([0] * 5, [0.2, 0.5, 1.0], [1, 2, 3, 4, 5])
    target = person([0.2, 0.9, 0.0, 0.5, 1.0], [0] * 3, [1] * 5)
    # ideal-point: 1*(1-0) + 2*(1-0.4) + 3*(1-1);  more-is-better: 4*0.5 + 5*1.0
    assert taste(viewer, target)[0, 0] == pytest.approx(1 + 1.2 + 0 + 2 + 5)


def test_probabilities_shape_and_range():
    world = build_world(CFG, seed=0)
    assert world.P.shape == (N, N) and world.Q.shape == (N, N)
    for M in (world.P, world.Q):
        assert 0 < M.min() and M.max() < 1


def test_quantile_threshold_overshoots_low_like_rates():
    """Why thresholds are calibrated: the sigmoid leaks probability past a quantile cut."""
    world = build_world(CFG, seed=0)
    s = scores(world.b, world.a, np.zeros((N, N)))
    thr = quantile_thresholds(s, world.b.like_rate)
    realized = (0.5 * (1 + np.tanh((s - thr[:, None]) / CFG.temp / 2))).mean(axis=1)
    assert realized.mean() > world.b.like_rate.mean() + 0.02


@pytest.mark.parametrize("side", ["a", "b"])
def test_calibrated_like_rates_hit_targets(side):
    world = build_world(CFG, seed=0)
    M, people = (world.P, world.a) if side == "a" else (world.Q, world.b)
    np.testing.assert_allclose(M.mean(axis=1), people.like_rate, atol=1e-4)


def test_pre_drawn_decisions_follow_probabilities():
    world = build_world(CFG, seed=0)
    assert world.DA.mean() == pytest.approx(world.P.mean(), abs=0.01)
    assert world.DB.mean() == pytest.approx(world.Q.mean(), abs=0.01)


def test_same_seed_same_world_across_w():
    lo, hi = build_world(with_w(CFG, 0.3), seed=0), build_world(with_w(CFG, 0.7), seed=0)
    np.testing.assert_array_equal(lo.a.traits, hi.a.traits)
    assert not np.array_equal(lo.P, hi.P)


def test_taste_is_mostly_personal():
    """Fitness and ambition are ranked the same way by everyone. If they dominate taste,
    'type' becomes a second universal score and w stops controlling concentration."""
    world = build_world(CFG, seed=0)
    v, t = world.a, world.b
    spread = [
        (v.importance[:, k, None] * (1 - np.abs(v.ideals[:, j, None] - t.traits[None, :, k])))
        .var(axis=1)
        .mean()
        for j, k in enumerate(IDEAL_POINT)
    ]
    spread += [
        (v.importance[:, k, None] * t.traits[None, :, k]).var(axis=1).mean() for k in MORE_IS_BETTER
    ]
    assert sum(spread[3:]) / sum(spread) < 0.3


def test_high_w_concentrates_likes_on_high_u():
    lo, hi = build_world(with_w(CFG, 0.1), seed=0), build_world(with_w(CFG, 0.9), seed=0)
    # Expected likes received = column sums: side B receives from A via P, A from B via Q.
    for received in (lambda w: w.P.sum(axis=0), lambda w: w.Q.sum(axis=0)):
        assert gini(received(hi)) > gini(received(lo)) + 0.1
    corr = [np.corrcoef(w.P.sum(axis=0), w.b.u)[0, 1] for w in (lo, hi)]
    assert corr[1] > 0.9 and corr[1] > corr[0] + 0.2


def test_like_probs_rows_are_viewers():
    world = build_world(CFG, seed=0)
    P = like_probs(world.a, world.b, np.zeros((N, N)), CFG.temp)
    assert P.shape == (world.a.n, world.b.n)
