"""Gale-Shapley stable matching, as a daily top pick (Hinge's old "Most Compatible").

One side proposes down its preference list; the other side holds its best offer so far and
rejects the rest; rejected proposers move to their next choice. It ends when nobody has
anyone left to propose to. The result is *stable*: no two people would both rather be with
each other than with who they got. It also favors the proposing side: each proposer gets the
best partner they could have in any stable matching.
"""

import numpy as np

from rankers.base import MarketState, Ranker


def gale_shapley(
    proposer_prefs: np.ndarray, receiver_prefs: np.ndarray, allowed: np.ndarray
) -> np.ndarray:
    """partner[p] = the receiver proposer p ends up with, or -1. Higher pref = preferred."""
    n_p, n_r = proposer_prefs.shape
    choices = np.argsort(-np.where(allowed, proposer_prefs, -np.inf), axis=1)
    n_choices = allowed.sum(axis=1)
    next_choice = np.zeros(n_p, int)
    held_by = np.full(n_r, -1)  # receiver -> proposer they're holding
    free = list(range(n_p))
    while free:
        p = free.pop()
        if next_choice[p] == n_choices[p]:
            continue  # proposed to everyone allowed; stays single
        r = choices[p, next_choice[p]]
        next_choice[p] += 1
        current = held_by[r]
        if current == -1:
            held_by[r] = p
        elif receiver_prefs[r, p] > receiver_prefs[r, current]:
            held_by[r] = p
            free.append(current)  # dumped: back to proposing
        else:
            free.append(p)  # rejected: try the next choice
    partner = np.full(n_p, -1)
    held = held_by >= 0
    partner[held_by[held]] = np.nonzero(held)[0]
    return partner


class GaleShapleyRanker(Ranker):
    """Each day, a stable matching over pairs still eligible on both sides, using the true
    preferences (side A proposes with P, side B accepts with Q). Each person's list is their
    one partner, so run it with list_len = 1."""

    name = "gale_shapley"

    def reset(self, world, rng) -> None:
        super().reset(world, rng)
        self.day = None  # the matching is computed once per day and shared by both sides

    def scores(self, side: str, state: MarketState) -> np.ndarray:
        if self.day != state.day:
            allowed = state.eligible["a"] & state.eligible["b"].T
            self.partner = gale_shapley(state.world.P, state.world.Q, allowed)
            self.day = state.day
        n = len(self.partner)
        s = np.full((n, n), -np.inf)  # unpaired people get no pick today
        a = np.nonzero(self.partner >= 0)[0]
        if side == "a":
            s[a, self.partner[a]] = 1
        else:
            s[self.partner[a], a] = 1
        return s
