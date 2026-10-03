import numpy as np
import pytest

from sim.config import FITNESS, SIDE_A, SIDE_B, Config, classroom, hinge_stages, with_w
from sim.population import make_side

N = 5000
CFG = Config()


def side(side_cfg=SIDE_A, cfg=CFG, seed=0):
    return make_side(side_cfg, cfg, N, np.random.default_rng(seed))


def test_shapes_and_ranges():
    s = side()
    assert s.u.shape == s.w.shape == s.like_rate.shape == s.attention.shape == (N,)
    assert s.traits.shape == s.importance.shape == (N, 5)
    assert s.ideals.shape == (N, 3)
    for arr in (s.w, s.traits, s.ideals):
        assert arr.min() >= 0 and arr.max() <= 1
    lo, hi = Config().like_rate_clip
    assert s.like_rate.min() >= lo and s.like_rate.max() <= hi
    assert s.attention.min() >= 5 and s.attention.max() <= 15
    assert s.importance.min() > 0


def test_same_seed_same_people():
    a, b = side(seed=1), side(seed=1)
    np.testing.assert_array_equal(a.traits, b.traits)
    np.testing.assert_array_equal(a.like_rate, b.like_rate)


def test_fitness_partly_reveals_u():
    s = side()
    assert np.corrcoef(s.u, s.traits[:, FITNESS])[0, 1] == pytest.approx(0.4, abs=0.04)


@pytest.mark.parametrize("side_cfg", [SIDE_A, SIDE_B])
def test_like_rate_mean_matches_side_target(side_cfg):
    a, b = side_cfg.like_rate_beta
    assert side(side_cfg).like_rate.mean() == pytest.approx(a / (a + b), abs=0.02)


def test_attractive_people_are_pickier():
    s = side()
    assert np.corrcoef(s.u, s.like_rate)[0, 1] < -0.05


def test_w_sweep_moves_the_same_people():
    lo = side(with_w(Config(), 0.4).side_a)
    hi = side(with_w(Config(), 0.5).side_a)
    unclipped = (lo.w > 0) & (hi.w < 1)
    np.testing.assert_allclose((hi.w - lo.w)[unclipped], 0.1)
    np.testing.assert_array_equal(lo.traits, hi.traits)


def test_with_w_keeps_the_side_gap():
    cfg = with_w(Config(), 0.8)
    assert cfg.side_a.w_mean == 0.8
    assert cfg.side_b.w_mean == pytest.approx(0.7)


def test_days_follow_exposure_targets():
    assert classroom(500).days == 11
    stages = hinge_stages(2000)
    assert {c.days for c in stages.values()} == {30}
