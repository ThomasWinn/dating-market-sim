"""True preferences: P[a, b] = chance A likes B, Q[b, a] = chance B likes A.

The chain for viewer A looking at target B (Q is the same with the sides swapped):

    taste(A→B) = Σ_{k ∈ ideal-point}    imp_A,k · (1 − |ideal_A,k − trait_B,k|)   my type
               + Σ_{k ∈ more-is-better} imp_A,k · trait_B,k                     fitter

    score(A→B) = w_A · z(u)_B  +  (1 − w_A) · z_row(taste)_A,B  +  chem(A, B)

        z(u)      u standardized over all targets (mean 0, SD 1)
        z_row     taste standardized within A's row, so every viewer's fit has SD 1
        chem      Normal(0, σ_c²), one draw per pair, shared: chem(B, A) = chem(A, B)

    P[A, B]    = sigmoid((score(A→B) − θ_A) / T),      sigmoid(x) = 1 / (1 + e^−x)
    θ_A        = the value with  mean over B of P[A, B] = like_rate_A   (bisection, per row)
    D_A[A, B]  = 1 if U(0, 1) < P[A, B]                the decision itself, drawn once

Symbols -> config: T = temp, σ_c = chemistry_sd. w, imp, ideal, u, like_rate come from
population.py.

How the knobs interact (measured at 500/side unless noted):
- w: share of weight on universal appeal. Raising side A's average from 0.1 to 0.9 raises the
  Gini of expected likes received (A→B) from 0.27 to 0.42, and likes start tracking u
  (corr 0.38 -> 0.97). Both parts are z-scored first, so w = 0.6 really means 60% appeal.
- imp on more-is-better traits: everyone ranks fitness and ambition the same way, so they act
  like a second u inside "type". At the plan's weights they were 78% of taste variance and w
  barely changed concentration; at the current weights they're about 18%.
- T: how sharp a decision is. As T → 0, P becomes a hard 0/1 cutoff at θ. A large T blurs
  decisions toward a coin flip near θ. T is also why a plain quantile θ overshoots: at
  T = 0.3, side B's 10.3% target came out at 14.5% before calibration.
- like_rate only moves the bar θ. It never changes who ranks above whom for a viewer.
- σ_c: one-off clicks no trait predicts. Because it's shared, mutual interest is a little
  more likely than independent draws would give.

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


SIDES = ("a", "b")
OTHER = {"a": "b", "b": "a"}
SIDE_ID = {"a": 0, "b": 1}


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

    def probs(self, side: str) -> np.ndarray:
        """Like chances oriented [viewer on `side`, candidate on the other side]."""
        return self.P if side == "a" else self.Q


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
