"""Shared chart styling. Colors follow the ranker (never its rank), in a fixed validated order;
lines are 2px with a ~10% wash for the min-max band across seeds; grid and axes stay recessive.
Every chart has a CSV twin in results/ for exact values."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)

# Fixed categorical slots (validated: worst adjacent CVD dE 9.1, normal-vision dE 19.6).
RANKER_COLORS = {
    "random": "#2a78d6",
    "popularity": "#eb6834",
    "elo": "#1baf7a",
    "one_sided_oracle": "#eda100",
    "reciprocal_oracle": "#e87ba4",
    "gale_shapley": "#008300",
}
LABELS = {
    "random": "Random",
    "popularity": "Popularity",
    "elo": "Elo",
    "one_sided_oracle": "One-sided oracle",
    "reciprocal_oracle": "Reciprocal oracle",
    "gale_shapley": "Gale-Shapley",
}

# Sides are a different kind of entity from rankers, so they get their own pair
# (slots 7-8; validated: CVD dE 22.7, contrast >= 3:1).
SIDE_COLORS = {"a": "#4a3aa7", "b": "#e34948"}
LABELS |= {"a": "Side A (likes often)", "b": "Side B (picky)"}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    }
)


def series_lines(
    ax, df: pd.DataFrame, x: str, y: str, series: str = "ranker", colors: dict = RANKER_COLORS
) -> None:
    """One 2px line per series (mean over seeds) with a faint min-max band."""
    order = [s for s in colors if s in set(df[series])]
    for name in order:
        g = df[df[series] == name].groupby(x)[y].agg(["mean", "min", "max"]).sort_index()
        color = colors[name]
        ax.fill_between(g.index, g["min"], g["max"], color=color, alpha=0.12, linewidth=0)
        ax.plot(
            g.index,
            g["mean"],
            color=color,
            linewidth=2,
            label=LABELS[name],
            marker="o",
            markersize=5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
        )


def figure(ncols: int = 1, width: float = 7.0, height: float = 4.2):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), squeeze=False)
    return fig, axes[0]


def finish(fig, path: Path, title: str, subtitle: str, legend_ax=None) -> None:
    fig.suptitle(
        title, x=0.01, y=0.99, ha="left", va="top", fontsize=13, fontweight="semibold", color=INK
    )
    fig.text(0.01, 0.925, subtitle, ha="left", va="top", fontsize=9.5, color=INK_2)
    ax = legend_ax if legend_ax is not None else fig.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) >= 2:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(handles), 6),
            bbox_to_anchor=(0.5, 0.0),
            fontsize=9,
            handlelength=1.6,
            labelcolor=INK_2,
        )
    fig.tight_layout(rect=(0, 0.07, 1, 0.88))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
