"""True preferences: P[a, b] = chance A likes B, Q[b, a] = chance B likes A.

Everything is computed once per world, fully vectorized. Loops run over the 5 traits or over
bisection steps, never over pairs.
"""

from dataclasses import dataclass

import numpy as np

from sim.config import IDEAL_POINT, MORE_IS_BETTER, Config
from sim.population import Side, make_side

# Every random part of a run gets its own stream, so changing one part (say, the ranker)
# doesn't shift the draws behind another (say, the people).
STREAMS = ("pop_a", "pop_b", "chemistry", "decisions", "patience", "ranker")


def rng_for(seed: int, stream: str) -> np.random.Generator:
    child = np.random.SeedSequence(seed).spawn(len(STREAMS))[STREAMS.index(stream)]
    return np.random.default_rng(child)


def taste(viewers: Side, targets: Side) -> np.ndarray:
    """(n_viewers, n_targets): how well each target fits each viewer's type."""
    out = np.zeros((viewers.n, targets.n))
    for j, k in enumerate(IDEAL_POINT):
        closeness = 1 - np.abs(viewers.ideals[:, j, None] - targets.traits[None, :, k])
        out += viewers.importance[:, k, None] * closeness
    for k in MORE_IS_BETTER:
        out += viewers.importance[:, k, None] * targets.traits[None, :, k]
    return out


def _z(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    return (x - x.mean(axis=axis, keepdims=True)) / x.std(axis=axis, keepdims=True)


def scores(viewers: Side, targets: Side, chemistry: np.ndarray) -> np.ndarray:
    # Standardize both parts so that w really is the share of weight on general appeal.
    # Without this, whichever part has the larger raw spread would dominate regardless of w.
    appeal = _z(targets.u)[None, :]
    fit = _z(taste(viewers, targets), axis=1)
    w = viewers.w[:, None]
    return w * appeal + (1 - w) * fit + chemistry


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + np.tanh(x / 2))  # same as 1 / (1 + e^-x), without overflow warnings


def quantile_thresholds(score: np.ndarray, like_rate: np.ndarray) -> np.ndarray:
    """Per-row cut so that each viewer's top `like_rate` share of scores sit above it."""
    ordered = np.sort(score, axis=1)
    idx = ((1 - like_rate) * (score.shape[1] - 1)).astype(int)
    return ordered[np.arange(len(score)), idx]


def calibrated_thresholds(
    score: np.ndarray, like_rate: np.ndarray, temp: float, iters: int = 40
) -> np.ndarray:
    """Per-row threshold so that mean(sigmoid((score - thr) / temp)) equals like_rate exactly.

    The quantile cut alone overshoots: people just below it still get a decent like chance
    from the sigmoid, and for picky people there are many more of those than people above it.
    The mean like chance falls as the threshold rises, so bisection on each row finds the
    exact value. All rows are bisected at once.
    """
    lo = score.min(axis=1) - 20 * temp  # like chance ~1 here: too low
    hi = score.max(axis=1) + 20 * temp  # like chance ~0 here: too high
    for _ in range(iters):
        mid = (lo + hi) / 2
        too_many = _sigmoid((score - mid[:, None]) / temp).mean(axis=1) > like_rate
        lo = np.where(too_many, mid, lo)
        hi = np.where(too_many, hi, mid)
    return (lo + hi) / 2


def like_probs(viewers: Side, targets: Side, chemistry: np.ndarray, temp: float) -> np.ndarray:
    s = scores(viewers, targets, chemistry)
    thr = calibrated_thresholds(s, viewers.like_rate, temp)
    return _sigmoid((s - thr[:, None]) / temp)


@dataclass(frozen=True)
class World:
    """The people and their true preferences. Built once, then shared by every ranker."""

    cfg: Config
    seed: int
    a: Side
    b: Side
    P: np.ndarray  # (n_a, n_b) chance A likes B
    Q: np.ndarray  # (n_b, n_a) chance B likes A
    # Each person's answer about each other person, drawn once. Every view still has the true
    # probability, but looking twice gives the same answer, and every ranker faces the same
    # coin flips, so differences between rankers are about ranking, not luck.
    DA: np.ndarray  # (n_a, n_b) bool: A would like B
    DB: np.ndarray  # (n_b, n_a) bool: B would like A


def build_world(cfg: Config, seed: int) -> World:
    n = cfg.n_per_side
    a = make_side(cfg.side_a, cfg, n, rng_for(seed, "pop_a"))
    b = make_side(cfg.side_b, cfg, n, rng_for(seed, "pop_b"))

    # One-off chemistry, shared: if A and B click, it shows up in both directions.
    chemistry = rng_for(seed, "chemistry").normal(0, cfg.chemistry_sd, (n, n))
    P = like_probs(a, b, chemistry, cfg.temp)
    Q = like_probs(b, a, chemistry.T, cfg.temp)

    rng = rng_for(seed, "decisions")
    DA = rng.random((n, n)) < P
    DB = rng.random((n, n)) < Q
    return World(cfg, seed, a, b, P, Q, DA, DB)
