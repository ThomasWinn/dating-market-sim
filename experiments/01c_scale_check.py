"""Do the key results hold as the market grows? 500 / 1,000 / 2,000 per side.

    uv run python experiments/01c_scale_check.py

Days scale with n (fixed exposure share), so each user sees the same share of the other side
at every size. Per-user-per-day rates should be roughly size-free; the check that matters is
that the rankers' ordering doesn't flip.
"""

import argparse
import time
from pathlib import Path

from sim.config import classroom, hinge
from sim.plots import INK_2, figure, finish, series_lines
from sim.runner import run_grid

ROOT = Path(__file__).resolve().parent.parent / "results"
SIZES = [500, 1000, 2000]
MODES = {"classroom": classroom, "hinge": hinge}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    start = time.perf_counter()
    jobs = [
        ({"mode": m, "n": n, "days": make(n).days}, make(n))
        for m, make in MODES.items()
        for n in SIZES
    ]
    df = run_grid(jobs, range(args.seeds), workers=args.workers)
    df["matches_per_user_day"] = df.matches / (df.n * df.days)
    df["dead"] = 100 * df.dead_like_share
    ROOT.mkdir(exist_ok=True)
    df.to_csv(ROOT / "01c_scale_check.csv", index=False)

    for mode in MODES:
        part = df[df["mode"] == mode]
        fig, axes = figure(ncols=3, width=11.5, height=4.4)
        for ax, (col, title) in zip(
            axes,
            [
                ("matches_per_user_day", "Matches per user per day"),
                ("gini_matches", "Gini of matches"),
                ("dead", "Likes never reviewed (%)"),
            ],
        ):
            series_lines(ax, part, "n", col)
            ax.set_xscale("log")
            ax.set_xticks(SIZES, [f"{n:,}" for n in SIZES])
            ax.minorticks_off()
            ax.set(xlabel="Users per side", title=title)
            ax.title.set_color(INK_2)
        finish(
            fig,
            ROOT / f"01c_{mode}_scale.png",
            f"Scale check, {mode} mode: results hold from 500 to 2,000 per side",
            f"Default w (A 0.6, B 0.5) · days scale with size · mean of {args.seeds} seeds",
        )

        order = part.groupby(["n", "ranker"]).matches.mean().unstack()
        ranks = order.rank(axis=1, ascending=False).astype(int)
        same = (ranks.nunique() == 1).all()
        print(f"\n{mode}: ranker order by matches (1 = most)\n{ranks.to_string()}")
        print(f"{mode}: ordering identical at every size: {same}")
    print(f"\n{len(df)} runs, {time.perf_counter() - start:.0f}s -> results/01c_*.png")


if __name__ == "__main__":
    main()
