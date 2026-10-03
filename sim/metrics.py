"""Market metrics."""

import numpy as np


def gini(x: np.ndarray) -> float:
    """0 = everyone has the same amount; (n - 1) / n = one person has everything."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def by_decile(values: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Mean of `values` within each tenth of people sorted by `key` (lowest key first)."""
    order = np.argsort(key, kind="stable")
    return np.array([values[chunk].mean() for chunk in np.array_split(order, 10)])


def summarize(world, res) -> dict[str, float]:
    """One row of metrics for a finished run. Side-prefixed keys are per side (a_, b_)."""
    n = world.cfg.n_per_side
    m = res.matched
    browse, inbox = res.browse, res.inbox
    matches = {"a": m.sum(axis=1), "b": m.sum(axis=0)}
    # A like "died" if it was never reviewed and the pair never matched anyway.
    pair_matched = np.where(
        inbox["side"] == 0,
        m[inbox["recipient"], inbox["sender"]],
        m[inbox["sender"], inbox["recipient"]],
    )
    dead = (inbox["day_reviewed"] < 0) & ~pair_matched

    out = {
        "matches": int(m.sum()),
        "gini_matches": gini(np.concatenate([matches["a"], matches["b"]])),
        "zero_match_pct": 100 * float(np.mean(np.concatenate([matches["a"], matches["b"]]) == 0)),
        "dead_like_share": float(dead.mean()) if len(dead) else 0.0,
    }
    for i, s in enumerate(("a", "b")):
        people = world.a if s == "a" else world.b
        mine = browse["side"] == i
        views, likes = int(mine.sum()), int(browse["liked"][mine].sum())
        received = np.bincount(
            browse["shown"][(browse["side"] == 1 - i) & browse["liked"]], minlength=n
        )
        to_me = inbox["side"] == i
        # A browse session is one person's day; did it end because they used every like?
        session = browse["day"][mine] * n + browse["viewer"][mine]
        likes_per_session = np.bincount(session, weights=browse["liked"][mine])
        active = np.bincount(session) > 0
        cap_hit = likes_per_session[active] >= world.cfg.like_cap
        out |= {
            f"{s}_gini_matches": gini(matches[s]),
            f"{s}_zero_match_pct": 100 * float(np.mean(matches[s] == 0)),
            f"{s}_gini_likes_received": gini(received),
            f"{s}_like_rate": likes / views if views else 0.0,
            f"{s}_likes_sent_per_user": likes / n,
            f"{s}_views_per_user_day": views / (n * res.days),
            f"{s}_dead_like_share": float(dead[to_me].mean()) if to_me.any() else 0.0,
            f"{s}_exposure": float(res.exposure[s].mean()),
            f"{s}_cap_hit_share": float(cap_hit.mean()) if len(cap_hit) else 0.0,
        }
        for d, v in enumerate(by_decile(matches[s], people.u)):
            out[f"{s}_u_decile_{d}"] = float(v)
    return out
