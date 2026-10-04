"""Every number in the simulator lives here. All defaults are placeholders until calibration.

Asymmetric vs. symmetric is not a switch: it's just per-side numbers. `symmetric(cfg)` copies
side A onto side B. The only switches that change code paths are the Hinge-mode mechanics.
"""

from dataclasses import dataclass, replace
from typing import Literal

# Displayed traits, in column order. The first three are "closer to my ideal is better";
# the last two are "more is better".
TRAITS = ("intent", "travel", "religiosity", "fitness", "ambition")
IDEAL_POINT = (0, 1, 2)
MORE_IS_BETTER = (3, 4)
FITNESS = 3


@dataclass(frozen=True)
class SideConfig:
    # Formulas using these are in population.py and preferences.py.
    w_mean: float  # w̄: average weight on appeal u. Higher = everyone wants the same people
    like_rate_beta: tuple[float, float]  # (a, b): mean like-rate a / (a + b), before the u shift
    importance: tuple[float, float, float, float, float]  # avg weight per trait (TRAITS order).
    # Big weights on fitness / ambition act like a second universal score (see note below).
    w_spread: float = 0.15  # s: each person's w is w̄ ± up to s
    attention: tuple[int, int] = (5, 15)  # likes read per day. Lower = more likes die unread


# Side A (men) likes more often and weights looks more; side B (women) is pickier and weights ambition more.
#
# The plan's importances were (1, 1, 1, 2, 1) for A and (1, 1, 1, 1.8, 1.3) for B. Those made
# 78% of each person's taste variance come from fitness and ambition, which everyone ranks the
# same way, so "type" acted as a second universal score and the w sweep came out flat (Gini of
# likes received 0.38 at w = 0.1 vs 0.41 at w = 0.9). The two more-is-better weights below
# keep the same pattern, scaled down 4x, which brings that share to about 18%.
SIDE_A = SideConfig(w_mean=0.6, like_rate_beta=(2, 3), importance=(1.0, 1.0, 1.0, 0.5, 0.25))
SIDE_B = SideConfig(w_mean=0.5, like_rate_beta=(4, 36), importance=(1.0, 1.0, 1.0, 0.45, 0.325))


@dataclass(frozen=True)
class Config:
    n_per_side: int = 500
    days: int = 12
    side_a: SideConfig = SIDE_A
    side_b: SideConfig = SIDE_B

    # Population (formulas in population.py)
    ideal_self_similarity: float = 0.5  # ρ: ideal = ρ·own + (1 − ρ)·random. Higher = "like me"
    importance_jitter: float = 0.3  # each person's importance is the side average × U(0.7, 1.3)
    fitness_u_corr: float = 0.4  # c: higher = fitness reveals more of hidden appeal u
    like_rate_u_shift: float = -0.05  # δ per SD of u. More negative = attractive people pickier
    like_rate_clip: tuple[float, float] = (0.02, 0.9)

    # True preferences (formulas in preferences.py)
    temp: float = 0.3  # T: lower = sharper yes/no, higher = closer to a coin flip near the bar
    chemistry_sd: float = 0.2  # σ_c: higher = more one-off clicks that no trait predicts

    # Browsing. Classroom mode: look at all 10. Hinge mode: scroll up to 50, may quit early.
    list_len: int = 10
    quit_prob: float = 0.0  # q: chance of quitting after each profile. Avg views ≈ 1/q
    like_cap: int = 8  # likes per day; liking back is free. Binds side A on 35% of Hinge days

    # Inbox
    inbox_order: Literal["best", "newest"] = "best"  # "newest" cost reciprocal 10% of matches
    carry_over: bool = False  # keep unread likes forever. Cost reciprocal another 12%, because a
    # stale like also hides that pair from each other's feeds

    # Rankers
    explore_frac: float = 0.1  # share of popularity / Elo slots filled at random (every 10th)
    elo_start: float = 1500.0
    elo_k: float = 32.0  # K: most a rating can move per swipe. Higher = faster but noisier


# Exposure targets: the share of the other side an average user sees while browsing over a
# whole run. Days are derived from these so results stay comparable across market sizes.
# Inbox reviews add more on top (about 7 points for side B), so the classroom target leaves room.
CLASSROOM_EXPOSURE = 0.22
HINGE_EXPOSURE = 0.27  # leaves room for inbox reviews, as above
HINGE_VIEWS_PER_DAY = 20  # ~1 / quit_prob
HINGE_LIST_LEN = 50
HINGE_QUIT_PROB = 0.05


def days_for_exposure(n_per_side: int, views_per_day: float, exposure: float) -> int:
    return max(1, int(exposure * n_per_side / views_per_day))


def classroom(n_per_side: int = 500) -> Config:
    days = days_for_exposure(n_per_side, Config.list_len, CLASSROOM_EXPOSURE)
    return Config(n_per_side=n_per_side, days=days)


def hinge_stages(n_per_side: int = 500) -> dict[str, Config]:
    """Classroom mode, then each Hinge mechanic added on top of the previous one.

    Days are held fixed across stages (set by the most-exposing stage) so that each
    stage differs from the previous one by exactly one mechanic.
    """
    days = days_for_exposure(n_per_side, HINGE_VIEWS_PER_DAY, HINGE_EXPOSURE)
    base = replace(classroom(n_per_side), days=days)
    scrolling = replace(base, list_len=HINGE_LIST_LEN, quit_prob=HINGE_QUIT_PROB)
    newest_first = replace(scrolling, inbox_order="newest")
    carry_over = replace(newest_first, carry_over=True)
    return {
        "classroom": base,
        "+scrolling": scrolling,
        "+newest_first": newest_first,
        "+carry_over": carry_over,
    }


def hinge(n_per_side: int = 500) -> Config:
    return hinge_stages(n_per_side)["+carry_over"]


def with_w(cfg: Config, a_mean: float) -> Config:
    """Move both sides' average w together, keeping the gap between them (the w sweep)."""
    gap = cfg.side_a.w_mean - cfg.side_b.w_mean
    return replace(
        cfg,
        side_a=replace(cfg.side_a, w_mean=a_mean),
        side_b=replace(cfg.side_b, w_mean=a_mean - gap),
    )


def symmetric(cfg: Config) -> Config:
    return replace(cfg, side_b=cfg.side_a)
