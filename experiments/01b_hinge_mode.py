"""Add Hinge mechanics one at a time and measure what each one changes.

    uv run python experiments/01b_hinge_mode.py            # 500/side
    uv run python experiments/01b_hinge_mode.py --n 2000

Stages are cumulative: classroom -> +scrolling -> +newest_first -> +carry_over. Days are the
same for every stage, so each step differs from the previous one by exactly one mechanic.
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from sim.config import hinge_stages
from sim.plots import INK_2, SIDE_COLORS, figure, finish, series_lines
from sim.runner import run_grid

ROOT = Path(__file__).resolve().parent.parent / "results"
STAGE_LABELS = ["Classroom", "+ Scrolling", "+ Newest-first", "+ Carry-over"]


def stage_axis(ax, title: str) -> None:
    ax.set_xticks(range(len(STAGE_LABELS)), STAGE_LABELS, rotation=20, ha="right")
    ax.set_title(title, color=INK_2)


def charts(df: pd.DataFrame, n: int, seeds: int, days: int, out: Path) -> None:
    sub = (
        f"Each step adds one mechanic · {n:,} per side · {days} days · "
        f"mean of {seeds} seeds, band = min–max"
    )
    df = df.assign(dead=100 * df.dead_like_share)

    fig, axes = figure(ncols=3, width=11.5, height=4.6)
    for ax, (col, title) in zip(
        axes,
        [
            ("matches", "Total matches"),
            ("dead", "Likes never reviewed (%)"),
            ("zero_match_pct", "Users with zero matches (%)"),
        ],
    ):
        series_lines(ax, df, "stage_i", col)
        stage_axis(ax, title)
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    finish(fig, out / "01b_outcomes_by_stage.png", "What each Hinge mechanic does to outcomes", sub)

    # The emergent split between the sides, shown for the random ranker so it's about
    # the people, not the ranking.
    rows = []
    for side in "ab":
        part = df[df.ranker == "random"][["stage_i", "seed"]].copy()
        part["side"] = side
        r = df[df.ranker == "random"]
        part["views"] = r[f"{side}_views_per_user_day"].values
        part["likes"] = r[f"{side}_likes_sent_per_user"].values / days
        part["cap"] = 100 * r[f"{side}_cap_hit_share"].values
        part["dead_to"] = 100 * r[f"{side}_dead_like_share"].values
        rows.append(part)
    by_side = pd.concat(rows)
    fig, axes = figure(ncols=4, width=13, height=4.6)
    for ax, (col, title) in zip(
        axes,
        [
            ("views", "Profiles viewed per day"),
            ("likes", "Likes sent per day"),
            ("cap", "Days ending at the like cap (%)"),
            ("dead_to", "Likes to this side that died (%)"),
        ],
    ):
        series_lines(ax, by_side, "stage_i", col, series="side", colors=SIDE_COLORS)
        stage_axis(ax, title)
    finish(
        fig,
        out / "01b_sides_by_stage.png",
        "Same app, different behavior: the like-happy side hits the cap, the picky side "
        "runs out of patience",
        sub + " · random ranker",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500, help="users per side")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    start = time.perf_counter()
    stages = hinge_stages(args.n)
    jobs = [({"stage": name, "stage_i": i}, cfg) for i, (name, cfg) in enumerate(stages.items())]
    df = run_grid(jobs, range(args.seeds), workers=args.workers)
    out = ROOT / f"n{args.n}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "01b_hinge_stages.csv", index=False)
    days = next(iter(stages.values())).days
    charts(df, args.n, args.seeds, days, out)

    cols = [
        "matches",
        "dead_like_share",
        "zero_match_pct",
        "a_views_per_user_day",
        "b_views_per_user_day",
        "a_cap_hit_share",
        "b_cap_hit_share",
    ]
    table = df.groupby(["stage_i", "stage", "ranker"])[cols].mean().round(3)
    print(table.to_string())
    print(f"{len(df)} runs, {time.perf_counter() - start:.0f}s -> results/n{args.n}/01b_*.png")


if __name__ == "__main__":
    main()
