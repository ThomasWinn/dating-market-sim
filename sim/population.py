"""Generate one side of the market: displayed traits plus the hidden variables behind them.

For each person i on a side (every draw independent across people):

    u_i          ~ Normal(0, 1)                                  general appeal (hidden)
    w_i          = clip(w̄ + s · U(-1, 1), 0, 1)                  looks-vs-type weight (hidden)
    trait_i,k    ~ U(0, 1)                                       intent, travel, religiosity,
                                                                 ambition (displayed)
    fitness_i    = rank(c · u_i + √(1 − c²) · ε_i) / n           ε ~ N(0, 1) (displayed)
    ideal_i,k    = ρ · trait_i,k + (1 − ρ) · U(0, 1)             ideal-point traits (hidden)
    imp_i,k      = imp̄_k · U(1 − j, 1 + j)                       care about trait k (hidden)
    like_rate_i  = clip(Beta(a, b) + δ · u_i, lo, hi)            pickiness (hidden)
    attention_i  ~ integer in [lo, hi]                           likes reviewed/day (hidden)

Symbols -> config: w̄ = SideConfig.w_mean, s = w_spread, c = fitness_u_corr,
ρ = ideal_self_similarity, imp̄ = SideConfig.importance, j = importance_jitter,
(a, b) = like_rate_beta, δ = like_rate_u_shift, (lo, hi) = like_rate_clip / attention.

How the pieces interact:
- c: the latent c·u + √(1 − c²)·ε has corr exactly c with u, and ranking it keeps the order
  while making fitness uniform like the other traits. Measured corr(u, fitness) = 0.40.
  A displayed trait that leaks hidden appeal is what lets a model learn u from profiles.
- (a, b), δ: E[like_rate] = a / (a + b) = 0.40 on side A, 0.10 on side B. The δ·u shift
  averages out but ties pickiness to appeal: corr(u, like_rate) = −0.24 on A and −0.70 on B,
  because B's Beta is so narrow (SD ≈ 0.047) that the shift dominates its spread.
- ρ: Var(ideal) = (ρ² + (1 − ρ)²) / 12, which is smallest at ρ = 0.5. Ideals then bunch
  toward the middle, so the ideal-point traits barely separate one candidate from another (see
  `taste` in preferences.py). Higher ρ also means "I want someone like me".
- s: the U(-1, 1) offset is drawn once per person, so sweeping w̄ shifts the same people.
"""

from dataclasses import dataclass

import numpy as np

from sim.config import FITNESS, IDEAL_POINT, Config, SideConfig


@dataclass(frozen=True)
class Side:
    u: np.ndarray  # (n,) general appeal. Hidden: users react to it, models never see it
    w: np.ndarray  # (n,) looks-vs-type weight. Hidden
    traits: np.ndarray  # (n, 5) displayed on the profile, in [0, 1]
    ideals: np.ndarray  # (n, 3) hidden ideal value for each ideal-point trait
    importance: np.ndarray  # (n, 5) hidden weight on each trait
    like_rate: np.ndarray  # (n,) hidden target share of profiles this person likes
    attention: np.ndarray  # (n,) hidden number of incoming likes reviewed per day

    @property
    def n(self) -> int:
        return len(self.u)


def _uniform_ranks(x: np.ndarray) -> np.ndarray:
    """Map values to (0, 1) by rank, so the result is uniform but keeps x's ordering."""
    return (np.argsort(np.argsort(x)) + 0.5) / len(x)


def make_side(side_cfg: SideConfig, cfg: Config, n: int, rng: np.random.Generator) -> Side:
    u = rng.standard_normal(n)

    # Drawing the offset once (instead of drawing w directly) means a w sweep moves the
    # same people up and down, rather than re-shuffling who is looks-driven.
    w = np.clip(side_cfg.w_mean + side_cfg.w_spread * rng.uniform(-1, 1, n), 0, 1)

    traits = rng.uniform(0, 1, (n, 5))
    c = cfg.fitness_u_corr
    traits[:, FITNESS] = _uniform_ranks(c * u + np.sqrt(1 - c**2) * rng.standard_normal(n))

    rho = cfg.ideal_self_similarity
    ideals = rho * traits[:, IDEAL_POINT] + (1 - rho) * rng.uniform(0, 1, (n, len(IDEAL_POINT)))

    jitter = rng.uniform(1 - cfg.importance_jitter, 1 + cfg.importance_jitter, (n, 5))
    importance = np.asarray(side_cfg.importance) * jitter

    a, b = side_cfg.like_rate_beta
    like_rate = np.clip(rng.beta(a, b, n) + cfg.like_rate_u_shift * u, *cfg.like_rate_clip)

    lo, hi = side_cfg.attention
    attention = rng.integers(lo, hi + 1, n)

    return Side(u, w, traits, ideals, importance, like_rate, attention)
