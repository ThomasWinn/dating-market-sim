"""Generate one side of the market: displayed traits plus the hidden variables behind them."""

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
