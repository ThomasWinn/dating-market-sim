"""Every baseline ranker x the w sweep x seeds.

    uv run python experiments/01_baselines_sweep.py               # 500/side, both modes
    uv run python experiments/01_baselines_sweep.py --n 2000      # final numbers

w on the x-axis is side A's average looks-vs-type weight (side B stays 0.1 lower): how much
attraction is universal. It plays the same role as lambda in the MODE paper.
"""

import argparse
import time
from pathlib import Path

from sim.config import classroom, hinge, with_w
from sim.plots import INK_2, figure, finish, series_lines
from sim.runner import run_grid

RESULTS = Path(__file__).resolve().parent.parent / "results"
W_VALUES = [round(0.1 * i, 1) for i in range(1, 10)]
HIGH_W = 0.8
MODES = {"classroom": classroom, "hinge": hinge}


def charts(df, mode: str, n: int, seeds: int) -> None:
    sub = f"{mode.capitalize()} mode · {n:,} per side · mean of {seeds} seeds, band = min–max"
    prefix = RESULTS / f"01_{mode}"

    fig, (ax,) = figure()
    series_lines(ax, df, "w", "matches")
    ax.set(xlabel="w (side A average weight on general appeal)", ylabel="Total matches")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    finish(fig, Path(f"{prefix}_matches_vs_w.png"), "Matches vs. how universal attraction is", sub)

    fig, (ax,) = figure()
    series_lines(ax, df, "w", "gini_matches")
    ax.set(xlabel="w (side A average weight on general appeal)", ylabel="Gini of matches per user")
    finish(fig, Path(f"{prefix}_gini_vs_w.png"), "Inequality of matches vs. w", sub)

    fig, (ax,) = figure()
    series_lines(ax, df.assign(dead=100 * df.dead_like_share), "w", "dead")
    ax.set(
        xlabel="w (side A average weight on general appeal)",
        ylabel="Likes never reviewed (% of likes sent)",
    )
    finish(
        fig,
        Path(f"{prefix}_dead_likes_vs_w.png"),
        "Congestion: likes that died unread in an inbox",
        sub,
    )

    # Who gets left behind: matches per user by hidden-appeal decile, at high w.
    high = df[df.w == HIGH_W]
    fig, axes = figure(ncols=2, width=9)
    for ax, side in zip(axes, "ab"):
        cols = [f"{side}_u_decile_{d}" for d in range(10)]
        long = high.melt(id_vars=["ranker", "seed"], value_vars=cols, var_name="decile")
        long["decile"] = long.decile.str.rsplit("_", n=1).str[1].astype(int) + 1
        series_lines(ax, long, "decile", "value")
        ax.set(
            xlabel="Hidden appeal decile (1 = lowest)",
            xticks=range(1, 11),
            title=f"Side {side.upper()}",
        )
        ax.title.set_color(INK_2)
    axes[0].set_ylabel("Matches per user")
    finish(fig, Path(f"{prefix}_u_deciles.png"), f"Who gets left behind (w = {HIGH_W})", sub)

    # Why reciprocal recommendation exists: the two oracles side by side.
    oracles = df[df.ranker.isin(["one_sided_oracle", "reciprocal_oracle"])]
    oracles = oracles.assign(dead=100 * oracles.dead_like_share)
    fig, axes = figure(ncols=3, width=11)
    for ax, (col, label) in zip(
        axes,
        [
            ("matches", "Total matches"),
            ("dead", "Likes never reviewed (%)"),
            ("zero_match_pct", "Users with zero matches (%)"),
        ],
    ):
        series_lines(ax, oracles, "w", col)
        ax.set(xlabel="w", title=label)
        ax.title.set_color(INK_2)
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    finish(
        fig,
        Path(f"{prefix}_oracles.png"),
        'One-sided vs. reciprocal oracle: "will they like me back?" matters',
        sub,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500, help="users per side")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--mode", choices=["classroom", "hinge", "both"], default="both")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    for mode in ["classroom", "hinge"] if args.mode == "both" else [args.mode]:
        start = time.perf_counter()
        base = MODES[mode](args.n)
        jobs = [({"mode": mode, "w": w}, with_w(base, w)) for w in W_VALUES]
        df = run_grid(jobs, range(args.seeds), workers=args.workers)
        RESULTS.mkdir(exist_ok=True)
        df.to_csv(RESULTS / f"01_{mode}_sweep.csv", index=False)
        charts(df, mode, args.n, args.seeds)
        print(
            f"{mode}: {len(df)} runs, {base.days} days, {time.perf_counter() - start:.0f}s "
            f"-> results/01_{mode}_*.png"
        )


if __name__ == "__main__":
    main()
