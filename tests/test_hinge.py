"""Hinge-mode mechanics, one at a time: scrolling, newest-first inbox, carry-over."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from rankers.baselines import RandomRanker
from sim.config import Config, hinge_stages
from sim.market import Inbox, Market, simulate
from sim.metrics import summarize
from sim.preferences import build_world

N = 300


def market_with_inbox(cfg, senders, day_sent):
    """A market where side-B person 0 has pending likes from the given side-A senders."""
    world = build_world(cfg, seed=0)
    m = Market(world, RandomRanker())
    k = len(senders)
    m.inbox["b"] = Inbox(np.zeros(k, int), np.array(senders), np.array(day_sent))
    return world, m


def reviewed_senders(m):
    log = pd.DataFrame(m.log._join(m.log.inbox))
    return set(log[log.day_reviewed >= 0].sender)


def test_best_first_reviews_the_likes_you_would_like_most():
    cfg = Config(n_per_side=N, inbox_order="best")
    world, m = market_with_inbox(cfg, senders=list(range(40)), day_sent=[0] * 40)
    m.review(day=1)
    budget = world.b.attention[0]
    best = set(np.argsort(-world.Q[0, :40])[:budget])
    assert reviewed_senders(m) == best


def test_newest_first_reviews_the_latest_likes():
    cfg = Config(n_per_side=N, inbox_order="newest")
    senders = list(range(40))
    day_sent = [i // 10 for i in senders]  # 10 likes from each of days 0..3
    world, m = market_with_inbox(cfg, senders, day_sent)
    m.review(day=4)
    budget = world.b.attention[0]  # 5..15, so day 3's likes go first, then day 2's
    reviewed = reviewed_senders(m)
    assert len(reviewed) == budget
    assert min(day_sent[s] for s in reviewed) >= 2
    assert set(range(30, 40)) <= reviewed or budget < 10


@pytest.mark.parametrize("carry_over", [False, True])
def test_unreviewed_likes_vanish_or_carry_over(carry_over):
    cfg = Config(n_per_side=N, carry_over=carry_over)
    world, m = market_with_inbox(cfg, senders=list(range(40)), day_sent=[0] * 40)
    m.review(day=1)
    m.end_of_day(drop=not carry_over)
    left = 40 - world.b.attention[0]
    assert len(m.inbox["b"]) == (left if carry_over else 0)


def test_carry_over_lets_old_likes_be_reviewed_late():
    cfg = replace(hinge_stages(N)["+carry_over"], n_per_side=N)
    res = simulate(build_world(cfg, seed=0), RandomRanker())
    inbox = pd.DataFrame(res.inbox)
    reviewed = inbox[inbox.day_reviewed >= 0]
    assert (reviewed.day_reviewed - reviewed.day_sent > 1).any()


@pytest.fixture(scope="module")
def scrolling():
    cfg = replace(hinge_stages(N)["+scrolling"], n_per_side=N)
    rows = [
        summarize(w, simulate(w, RandomRanker())) for w in (build_world(cfg, s) for s in range(3))
    ]
    return pd.DataFrame(rows).mean()


def test_scrolling_like_happy_side_hits_the_cap(scrolling):
    assert scrolling.a_cap_hit_share > 5 * scrolling.b_cap_hit_share
    assert scrolling.a_cap_hit_share > 0.2


def test_scrolling_picky_side_runs_out_of_patience_first(scrolling):
    # Side B views more profiles per day (it rarely reaches the cap), yet sends far fewer likes.
    assert scrolling.b_views_per_user_day > scrolling.a_views_per_user_day
    assert scrolling.a_likes_sent_per_user > 2 * scrolling.b_likes_sent_per_user


def test_hinge_stages_keep_exposure_under_30_percent():
    for name, cfg in hinge_stages(500).items():
        res = simulate(build_world(cfg, seed=0), RandomRanker())
        for side in "ab":
            assert res.exposure[side].mean() <= 0.30, (name, side)
