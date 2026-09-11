"""Boxplots of PEHE across 20 training draws, grouped by training size.

One figure per dataset (plus a 2x2 contact sheet), x = training size, y = PEHE,
three boxes per size -- one per method. Written to figures/boxplot_*.png; the
single-shot line plots are left alone.

The two model-free reference levels from the line plots are carried over, since
they make each panel self-interpreting: above predict-zero is no signal, the
shaded band is level-only, below the oracle is real heterogeneity.

Draws that could not be fit are excluded and disclosed in a footnote -- a
uniform draw from Lalonde CPS (1.1% treated) sometimes contains zero treated
rows, which no per-arm learner can handle.
"""
from __future__ import annotations

import argparse
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from common import DATASETS, FIG_SUFFIX, FIGURES_DIR, RESULTS_DIR, TRAIN_SIZES
from plot_style import AXIS, GRID, INK, MUTED, SURFACE, _add_reference, _thousands

ALL_METHODS = {
    "tabpfn_t": ("TabPFN v3 (T-learner)", "#2a78d6"),
    "causalpfn": ("CausalPFN", "#e34948"),
    "dopfn": ("Do-PFN", "#1baf7a"),
}
# Narrowed by --methods; load()/draw()/legend_handles() read this at call time.
METHODS = dict(ALL_METHODS)

REPS_RESULTS = RESULTS_DIR / "reps"

# The contact sheet shows Lalonde CPS on these instead of TRAIN_SIZES.
LARGE_SIZES = [2000, 4000, 6000, 8000, 10000]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def load(suffix: str = FIG_SUFFIX) -> tuple[dict, dict, dict]:
    """Per-cell PEHEs restricted to draws every method could fit.

    Returns scores[(method, dataset, n)] = [pehe, ...] over the COMMON set of
    reps, dropped[(dataset, n)] = number of draws excluded, and
    attempted[(dataset, n)] = number of draws tried.

    The restriction matters. A uniform draw from Lalonde CPS (1.1% treated)
    often has zero treated rows, and the methods disagree about what to do:
    CausalPFN raises (its treatment encoder never saw t=1), TabPFN's T-learner
    raises (an arm needs >= 2 rows), and Do-PFN silently returns a number
    because it never checks. Summarising each method over whatever it happened
    to survive would compare different subsets of draws -- and would penalise
    Do-PFN, since only its boxes would carry the hardest draws.
    """
    raw = {}
    for method in METHODS:
        path = REPS_RESULTS / f"{method}{suffix}.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing -- run run_reps.py --method {method}")
        for r in json.loads(path.read_text(encoding="utf-8")):
            raw[(r["method"], r["dataset"], r["n_train"], r["rep"])] = r.get("pehe")

    cells = {(d, n) for _, d, n, _ in raw}
    scores, dropped, attempted = {}, {}, {}
    for dataset, n in cells:
        reps = sorted({rep for _, d, nn, rep in raw if d == dataset and nn == n})
        # A rep counts as attempted only once EVERY method has a record for it.
        # Reps still queued in a partial run are simply not counted -- otherwise
        # an interim plot would report them as failures.
        computed = [
            rep for rep in reps if all((m, dataset, n, rep) in raw for m in METHODS)
        ]
        common = [
            rep
            for rep in computed
            if all(raw[(m, dataset, n, rep)] is not None for m in METHODS)
        ]
        attempted[(dataset, n)] = len(computed)
        dropped[(dataset, n)] = len(computed) - len(common)
        for method in METHODS:
            scores[(method, dataset, n)] = [raw[(method, dataset, n, rep)] for rep in common]
    return scores, dropped, attempted


def draw(ax, bundle, dataset: str, sizes: list[int], *, reference: bool) -> list[tuple]:
    """Draw one panel. `bundle` is a (scores, dropped, attempted) triple, and
    `sizes` the training sizes on the x-axis -- passed explicitly so the contact
    sheet can put Lalonde CPS on its large-n sweep while the other three panels
    stay on the small-n one. Returns (n, dropped, attempted) for the footnote."""
    scores, dropped, attempted = bundle
    n_methods = len(METHODS)
    group_w = 0.78
    box_w = group_w / n_methods * 0.86  # the slack becomes the surface gap
    notes = [
        (n, dropped.get((dataset, n), 0), attempted.get((dataset, n), 0))
        for n in sizes
        if dropped.get((dataset, n), 0)
    ]

    whiskers, fliers = [], []
    for j, (method, (label, color)) in enumerate(METHODS.items()):
        offset = (j - (n_methods - 1) / 2) * (group_w / n_methods)
        data, positions = [], []
        for i, n in enumerate(sizes):
            vals = scores.get((method, dataset, n), [])
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

    # Scale to the whiskers, not the fliers. On Lalonde CPS at n=2,000 the
    # T-learner's treated arm is fit on ~21 rows and occasionally extrapolates
    # to PEHE 27,656 -- three times the median. Letting that set the axis
    # flattens every box into a strip. The clipped points are disclosed below.
    if whiskers:
        w_lo, w_hi = min(whiskers), max(whiskers)
        pad = 0.08 * (w_hi - w_lo) or 1.0
        ax.set_ylim(w_lo - pad, w_hi + pad)

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{n:,}" if n >= 1000 else str(n) for n in sizes])
    ax.set_xlim(-0.6, len(sizes) - 0.4)
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)

    if reference:
        # x is categorical here, so anchor the reference labels to tick positions.
        _add_reference(ax, dataset, xspan=(0, len(sizes) - 1))

    # Count clipped fliers against the FINAL limits -- _add_reference may have
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
    return notes


def legend_handles():
    return [
        Line2D([0], [0], marker="s", markersize=9, linestyle="none",
               markerfacecolor=c, markeredgecolor="none", label=lbl)
        for lbl, c in METHODS.values()
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--no-reference", action="store_true")
    ap.add_argument(
        "--methods", nargs="*", default=list(ALL_METHODS),
        help="subset of methods to plot, e.g. --methods tabpfn_t causalpfn",
    )
    ap.add_argument(
        "--out-suffix", default="",
        help="appended to output filenames so a variant does not overwrite",
    )
    args = ap.parse_args()
    globals()["METHODS"] = {k: v for k, v in ALL_METHODS.items() if k in args.methods}
    out_suffix = FIG_SUFFIX + args.out_suffix
    selected = {k: v for k, v in DATASETS.items() if k in args.datasets}
    reference = not args.no_reference

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load()
    n_reps = max(bundle[2].values(), default=0)

    for dataset, display in selected.items():
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        notes = draw(ax, bundle, dataset, TRAIN_SIZES, reference=reference)
        ax.set_title(
            f"{display}   ({n_reps} training draws)",
            color=INK, fontsize=13, fontweight="bold", loc="left", pad=12,
        )
        ax.set_xlabel("Training data size", color=MUTED, fontsize=10)
        ax.set_ylabel("root-PEHE  (lower is better)", color=MUTED, fontsize=10)
        ax.legend(
            handles=legend_handles(), frameon=False, fontsize=9, labelcolor=MUTED,
            loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, columnspacing=2.0,
        )
        if notes:
            fig.text(
                0.5, -0.10,
                "draws excluded (no treated units, unfittable for >=1 method) -- "
                + ", ".join(f"n={n:,}: {miss}/{tot}" for n, miss, tot in notes),
                ha="center", color=MUTED, fontsize=7.5,
            )
        fig.tight_layout()
        out = FIGURES_DIR / f"boxplot_{dataset}{out_suffix}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")

    if len(selected) == len(DATASETS) and not FIG_SUFFIX:
        # Lalonde CPS is shown on its large-n sweep: at n<=500 its 1.1% treated
        # rate makes two thirds of the draws unfittable and leaves extreme
        # outliers that flatten the panel. The other three stay on small n.
        large = load("_large_n")
        panels = [
            (ds, disp, large if ds == "lalonde_cps" else bundle,
             LARGE_SIZES if ds == "lalonde_cps" else TRAIN_SIZES)
            for ds, disp in selected.items()
        ]
        fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
        all_notes = []
        for ax, (dataset, display, data, sizes) in zip(axes.ravel(), panels):
            all_notes += [
                (display, n, miss, tot)
                for n, miss, tot in draw(ax, data, dataset, sizes, reference=reference)
            ]
            suffix_note = "  (n = 2,000-10,000)" if dataset == "lalonde_cps" else ""
            ax.set_title(display + suffix_note, color=INK, fontsize=12,
                         fontweight="bold", loc="left", pad=8)
        for ax in axes[-1]:
            ax.set_xlabel("Training data size", color=MUTED, fontsize=10)
        for ax in axes[:, 0]:
            ax.set_ylabel("root-PEHE  (lower is better)", color=MUTED, fontsize=10)
        fig.legend(
            handles=legend_handles(), frameon=False, fontsize=10, labelcolor=MUTED,
            loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.005), columnspacing=2.5,
        )
        if all_notes:
            fig.text(
                0.5, -0.035,
                "draws excluded (no treated units, unfittable for >=1 method) -- "
                + ", ".join(f"{d} n={n:,}: {miss}/{tot}" for d, n, miss, tot in all_notes),
                ha="center", color=MUTED, fontsize=7.5,
            )
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        out = FIGURES_DIR / f"boxplot_all_datasets{out_suffix}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")

    # Companion table: median and IQR, the numbers behind the boxes.
    lines = [f"# PEHE across {n_reps} training draws (median [Q1, Q3])", ""]
    for dataset, display in selected.items():
        lines += [f"## {display}", ""]
        lines.append("| Method | " + " | ".join(f"n={n}" for n in TRAIN_SIZES) + " |")
        lines.append("|---" * (len(TRAIN_SIZES) + 1) + "|")
        for method, (label, _) in METHODS.items():
            cells = []
            for n in TRAIN_SIZES:
                v = bundle[0].get((method, dataset, n), [])
                if not v:
                    cells.append("--")
                    continue
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                f = (lambda x: f"{x:,.0f}") if med >= 1000 else (lambda x: f"{x:.3f}")
                cells.append(f"{f(med)} [{f(q1)}, {f(q3)}] (n={len(v)})")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    out = RESULTS_DIR / f"summary_boxplot{out_suffix}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()