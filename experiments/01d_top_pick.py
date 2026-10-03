"""Gale-Shapley as a daily top pick vs. the reciprocal oracle's and random's top-1.

    uv run python experiments/01d_top_pick.py

Everyone gets exactly one profile a day (list_len = 1). Gale-Shapley pairs people up so that
nobody is anyone else's pick too; the reciprocal oracle gives each person their single best
match chance, even if thousands of others got the same person.
"""

import argparse
import time
from dataclasses import replace
from pathlib import Path

from sim.config import classroom, with_w
from sim.plots import INK_2, figure, finish, series_lines
from sim.runner import run_grid

ROOT = Path(__file__).resolve().parent.parent / "results"
W_VALUES = [round(0.1 * i, 1) for i in range(1, 10)]
RANKERS = ["random", "reciprocal_oracle", "gale_shapley"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500, help="users per side")
    parser.add_argument("--days", type=int, default=30, help="one pick a day, for a month")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    start = time.perf_counter()
    base = replace(classroom(args.n), list_len=1, days=args.days)
    jobs = [({"w": w}, with_w(base, w)) for w in W_VALUES]
    df = run_grid(jobs, range(args.seeds), rankers=RANKERS, workers=args.workers)
    df["dead"] = 100 * df.dead_like_share
    out = ROOT / f"n{args.n}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "01d_top_pick.csv", index=False)

    fig, axes = figure(ncols=3, width=11.5, height=4.4)
    for ax, (col, title) in zip(
        axes,
        [
            ("matches", "Total matches"),
            ("dead", "Likes never reviewed (%)"),
            ("gini_matches", "Gini of matches"),
        ],
    ):
        series_lines(ax, df, "w", col)
        ax.set(xlabel="w", title=title)
        ax.title.set_color(INK_2)
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    finish(
        fig,
        out / "01d_top_pick.png",
        "One daily top pick: stable matching vs. best odds",
        f"{args.n:,} per side · 1 profile a day for {args.days} days · "
        f"mean of {args.seeds} seeds, band = min–max",
    )

    print(
        df.groupby(["ranker", "w"])[
            ["matches", "dead_like_share", "gini_matches", "zero_match_pct", "seconds"]
        ]
        .mean()
        .round(3)
        .to_string()
    )
    print(f"{len(df)} runs, {time.perf_counter() - start:.0f}s -> results/n{args.n}/01d_*.png")


if __name__ == "__main__":
    main()
