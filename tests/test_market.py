from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from rankers.baselines import RandomRanker
from sim.config import Config, classroom, symmetric
from sim.market import simulate
from sim.metrics import summarize
from sim.preferences import build_world

CFG = Config(n_per_side=200, days=10)


@pytest.fixture(scope="module")
def run():
    world = build_world(CFG, seed=0)
    res = simulate(world, RandomRanker())
    browse, inbox = res.to_frames()
    return world, res, browse, inbox


def test_nobody_is_shown_the_same_person_twice(run):
    _, _, browse, _ = run
    assert not browse.duplicated(["side", "viewer", "shown"]).any()


def test_nobody_is_shown_someone_waiting_in_their_inbox(run):
    _, _, browse, inbox = run
    # In classroom mode a like sent on day d sits in the inbox on day d + 1 only.
    waiting = inbox.assign(day=inbox.day_sent + 1)[["side", "recipient", "sender", "day"]]
    waiting.columns = ["side", "viewer", "shown", "day"]
    assert browse.merge(waiting, on=["side", "viewer", "shown", "day"]).empty


def test_nobody_is_shown_someone_they_already_reviewed(run):
    _, _, browse, inbox = run
    reviewed = inbox[inbox.day_reviewed >= 0][["side", "recipient", "sender", "day_reviewed"]]
    reviewed.columns = ["side", "viewer", "shown", "day_reviewed"]
    both = browse.merge(reviewed, on=["side", "viewer", "shown"])
    assert (both.day < both.day_reviewed).all()  # browsing earlier is fine; later is not


def test_daily_likes_respect_the_cap(run):
    _, _, browse, _ = run
    assert browse.groupby(["day", "side", "viewer"]).liked.sum().max() <= CFG.like_cap


def test_reviews_respect_attention_budget(run):
    world, _, _, inbox = run
    reviewed = inbox[inbox.day_reviewed >= 0]
    per_day = reviewed.groupby(["day_reviewed", "side", "recipient"]).size().reset_index(name="k")
    budget = np.where(
        per_day.side == 0,
        world.a.attention[per_day.recipient],
        world.b.attention[per_day.recipient],
    )
    assert (per_day.k <= budget).all()


def test_classroom_reviews_only_yesterdays_likes(run):
    _, _, _, inbox = run
    reviewed = inbox[inbox.day_reviewed >= 0]
    assert (reviewed.day_reviewed - reviewed.day_sent == 1).all()


def test_every_like_is_accounted_for_once(run):
    _, _, browse, inbox = run
    assert len(inbox) == browse.liked.sum()


def test_matches_are_mutual_likes(run):
    world, res, _, inbox = run
    a, b = np.nonzero(res.matched)
    assert len(a) > 0
    assert world.DA[a, b].all() and world.DB[b, a].all()
    # ...and each one came from a like-back in somebody's inbox.
    backs = inbox[inbox.liked_back]
    pairs = set(
        zip(
            np.where(backs.side == 0, backs.recipient, backs.sender),
            np.where(backs.side == 0, backs.sender, backs.recipient),
        )
    )
    assert pairs == set(zip(a, b))


def test_browse_log_marks_matches(run):
    _, res, browse, _ = run
    a = np.where(browse.side == 0, browse.viewer, browse.shown)
    b = np.where(browse.side == 0, browse.shown, browse.viewer)
    np.testing.assert_array_equal(browse.matched, res.matched[a, b])


def test_same_seed_same_run():
    world = build_world(CFG, seed=3)
    one, two = simulate(world, RandomRanker()), simulate(world, RandomRanker())
    np.testing.assert_array_equal(one.matched, two.matched)


@pytest.mark.parametrize("side", ["a", "b"])
def test_random_ranker_realizes_target_like_rates(side, run):
    world, res, _, _ = run
    people = world.a if side == "a" else world.b
    assert summarize(world, res)[f"{side}_like_rate"] == pytest.approx(
        people.like_rate.mean(), abs=0.03
    )


def test_symmetric_market_gives_symmetric_sides():
    cfg = symmetric(replace(CFG, n_per_side=300))
    worlds = (build_world(cfg, seed) for seed in range(3))
    rows = pd.DataFrame([summarize(w, simulate(w, RandomRanker())) for w in worlds]).mean()
    # Absolute tolerances in each metric's own units (zero_match_pct is in percent).
    tolerance = {
        "like_rate": 0.02,
        "views_per_user_day": 0.1,
        "gini_likes_received": 0.02,
        "gini_matches": 0.03,
        "zero_match_pct": 1.5,
        "dead_like_share": 0.015,
    }
    for metric, tol in tolerance.items():
        assert rows[f"a_{metric}"] == pytest.approx(rows[f"b_{metric}"], abs=tol), metric


def test_classroom_preset_keeps_exposure_under_30_percent():
    world = build_world(classroom(500), seed=0)
    res = simulate(world, RandomRanker())
    for side in "ab":
        assert res.exposure[side].mean() <= 0.30
