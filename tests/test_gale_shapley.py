from dataclasses import replace
from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from rankers.gale_shapley import GaleShapleyRanker, gale_shapley
from sim.config import Config
from sim.market import simulate
from sim.preferences import build_world


def blocking_pairs(partner, prop_prefs, recv_prefs, allowed):
    """Pairs who'd both rather be with each other than with whoever they ended up with."""
    holder = {r: p for p, r in enumerate(partner) if r >= 0}
    out = []
    for p, r in zip(*np.nonzero(allowed)):
        mine = partner[p]
        p_wants = mine < 0 or prop_prefs[p, r] > prop_prefs[p, mine]
        theirs = holder.get(r, -1)
        r_wants = theirs < 0 or recv_prefs[r, p] > recv_prefs[r, theirs]
        if p_wants and r_wants and mine != r:
            out.append((p, r))
    return out


@pytest.mark.parametrize("seed", range(20))
def test_result_is_stable(seed):
    rng = np.random.default_rng(seed)
    n = 12
    prop, recv = rng.random((n, n)), rng.random((n, n))
    allowed = rng.random((n, n)) < 0.7
    partner = gale_shapley(prop, recv, allowed)
    assert blocking_pairs(partner, prop, recv, allowed) == []
    matched = partner >= 0
    assert allowed[np.nonzero(matched)[0], partner[matched]].all()
    assert len(set(partner[matched])) == matched.sum()  # nobody is held twice


@pytest.mark.parametrize("seed", range(10))
def test_proposers_get_their_best_stable_partner(seed):
    """Brute force every perfect matching of a tiny market: among the stable ones, each
    proposer does at least as well as in any other (the proposer-optimal property)."""
    rng = np.random.default_rng(seed)
    n = 5
    prop, recv = rng.random((n, n)), rng.random((n, n))
    allowed = np.ones((n, n), bool)
    ours = gale_shapley(prop, recv, allowed)
    stable = [
        np.array(m)
        for m in permutations(range(n))
        if not blocking_pairs(np.array(m), prop, recv, allowed)
    ]
    for m in stable:
        assert (prop[np.arange(n), ours] >= prop[np.arange(n), m]).all()


def test_top_pick_pairs_people_with_each_other():
    cfg = replace(Config(n_per_side=200, days=5), list_len=1)
    res = simulate(build_world(cfg, seed=0), GaleShapleyRanker())
    browse = pd.DataFrame(res.browse)
    for _, day in browse.groupby("day"):
        a_sees = set(zip(day[day.side == 0].viewer, day[day.side == 0].shown))
        b_sees = set(zip(day[day.side == 1].shown, day[day.side == 1].viewer))
        assert a_sees == b_sees  # if A's top pick is B, B's top pick is A
        assert day[day.side == 0].shown.is_unique  # nobody is anyone else's top pick too
