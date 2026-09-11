"""Figure 1: root-PEHE vs. training-set size, one panel per dataset.

Each panel shows one box per method at each training size, over the draws every
method could fit. Lalonde CPS is shown on its large-n sweep: at n <= 500 its
1.06% treated rate makes most draws unfittable.

The y-axis is scaled to the whiskers rather than the outliers. On Lalonde CPS at
n = 2,000 the T-learner's treated arm is fit on ~21 rows and occasionally
extrapolates to a PEHE three times the median; letting that set the axis would
flatten every box. Clipped points are counted in the panel corner.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from .. import config
from ..config import EXPERIMENTS
from ..results import common_scores
from .style import AXIS, COLORS, GRID, INK, MUTED, SURFACE, add_reference, thousands


def draw_boxes(ax, scores, dataset: str, columns, labels, *, reference: bool = True) -> None:
    """Draw grouped boxplots. `columns` are the (n_train, sigma) cells on the
    x-axis, in order; `labels` are their tick labels."""
    methods = list(config.METHODS)
    n_methods = len(methods)
    group_w = 0.78
    box_w = group_w / n_methods * 0.86  # the slack becomes the surface gap

    whiskers, fliers = [], []
    for j, method in enumerate(methods):
        color = COLORS[method]
        offset = (j - (n_methods - 1) / 2) * (group_w / n_methods)
        data, positions = [], []
        for i, (n, sigma) in enumerate(columns):
            vals = scores.get((dataset, n, sigma), {}).get(method, [])
            if vals:
                data.append(vals)
                positions.append(i + offset)
        if not data:
            continue
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=box_w,
            patch_artist=True,
            showfliers=True,
            medianprops=dict(color=SURFACE, linewidth=1.6),
            whiskerprops=dict(color=color, linewidth=1.2),
            capprops=dict(color=color, linewidth=1.2),
            flierprops=dict(
                marker="o", markersize=3, markerfacecolor=color,
                markeredgecolor="none", alpha=0.75,
            ),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_linewidth(1.2)
            patch.set_alpha(0.85)
        for w in bp["whiskers"]:
            whiskers += list(w.get_ydata())
        for f in bp["fliers"]:
            fliers += list(f.get_ydata())

    if whiskers:
        w_lo, w_hi = min(whiskers), max(whiskers)
        pad = 0.08 * (w_hi - w_lo) or 1.0
        ax.set_ylim(w_lo - pad, w_hi + pad)

    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, len(columns) - 0.4)
    ax.yaxis.set_major_formatter(FuncFormatter(thousands))
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)

    if reference:
        # x is categorical, so anchor the reference labels to tick positions.
        add_reference(ax, dataset, xspan=(0, len(columns) - 1))

    # Count clipped fliers against the FINAL limits: add_reference may have
    # raised the top to fit a reference line, bringing points back into view.
    lo, hi = ax.get_ylim()
    beyond = [v for v in fliers if not lo <= v <= hi]
    if beyond:
        top = max(beyond)
        shown = f"{top:,.0f}" if top >= 1000 else f"{top:.2f}"
        ax.annotate(
            f"{len(beyond)} point{'s' if len(beyond) > 1 else ''} beyond axis (max {shown})",
            xy=(0.99, 0.99),
            xycoords="axes fraction",
            ha="right",
            va="top",
            color=MUTED,
            fontsize=7.5,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5),
        )


def legend_handles():
    return [
        Line2D([0], [0], marker="s", markersize=9, linestyle="none",
               markerfacecolor=COLORS[m], markeredgecolor="none", label=label)
        for m, label in config.METHODS.items()
    ]


def plot(out: Path | None = None) -> Path:
    small = common_scores(EXPERIMENTS["sample-size"])
    large = common_scores(EXPERIMENTS["sample-size-large"])

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    notes = []
    for ax, (dataset, display) in zip(axes.ravel(), config.DATASETS.items()):
        experiment = EXPERIMENTS["sample-size-large" if dataset == "lalonde_cps" else "sample-size"]
        scores, dropped, attempted = large if dataset == "lalonde_cps" else small
        sizes = experiment.train_sizes[dataset]
        columns = [(n, None) for n in sizes]
        draw_boxes(ax, scores, dataset, columns, [f"{n:,}" if n >= 1000 else str(n) for n in sizes])
        notes += [
            (display, n, dropped[(dataset, n, None)], attempted[(dataset, n, None)])
            for n in sizes
            if dropped.get((dataset, n, None))
        ]
        suffix = "  (n = 2,000-10,000)" if dataset == "lalonde_cps" else ""
        ax.set_title(display + suffix, color=INK, fontsize=12, fontweight="bold", loc="left", pad=8)
    for ax in axes[-1]:
        ax.set_xlabel("Training data size", color=MUTED, fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("root-PEHE  (lower is better)", color=MUTED, fontsize=10)
    fig.legend(
        handles=legend_handles(), frameon=False, fontsize=10, labelcolor=MUTED,
        loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.005), columnspacing=2.5,
    )
    if notes:
        fig.text(
            0.5, -0.035,
            "draws excluded (no treated units, unfittable for >=1 method) -- "
            + ", ".join(f"{d} n={n:,}: {miss}/{tot}" for d, n, miss, tot in notes),
            ha="center", color=MUTED, fontsize=7.5,
        )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = out or config.FIGURES_DIR / "fig1_sample_size.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out
