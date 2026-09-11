"""Boxplots of PEHE against covariate contamination level.

x = sigma (how badly the observed confounders are measured), y = PEHE, one box
per method at each level. Same reference lines as the training-size figures:
they depend only on the test set's true CATE, which contamination leaves
untouched, so they remain valid anchors here.

The x labels carry the reliability corr(observed, true) = 1/sqrt(1+sigma^2)
alongside sigma, since that is the quantity the two noise mechanisms are matched
on and the one that is actually interpretable.
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

from common import DATASETS, FIGURES_DIR, RESULTS_DIR
from plot_style import AXIS, GRID, INK, MUTED, SURFACE, _add_reference, _thousands

ALL_METHODS = {
    "tabpfn_t": ("TabPFN v3 (T-learner)", "#2a78d6"),
    "causalpfn": ("CausalPFN", "#e34948"),
    "dopfn": ("Do-PFN", "#1baf7a"),
}
METHODS = dict(ALL_METHODS)
RESULTS = RESULTS_DIR / "contamination"
SIGMAS = [0.01, 0.1, 1.0, 10.0]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def reliability(sigma):
    return 1.0 / np.sqrt(1.0 + sigma ** 2)


def load():
    """scores[(method, dataset, sigma)] over draws every method could fit."""
    raw = {}
    for m in METHODS:
        p = RESULTS / f"{m}.json"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing -- run run_contamination.py --method {m}")
        for r in json.loads(p.read_text(encoding="utf-8")):
            raw[(r["method"], r["dataset"], r["sigma"], r["rep"])] = r.get("pehe")
    cells = {(d, s) for _, d, s, _ in raw}
    scores, dropped, attempted = {}, {}, {}
    for ds, s in cells:
        reps = sorted({rep for _, d, ss, rep in raw if d == ds and ss == s})
        computed = [r for r in reps if all((m, ds, s, r) in raw for m in METHODS)]
        common = [r for r in computed if all(raw[(m, ds, s, r)] is not None for m in METHODS)]
        attempted[(ds, s)], dropped[(ds, s)] = len(computed), len(computed) - len(common)
        for m in METHODS:
            scores[(m, ds, s)] = [raw[(m, ds, s, r)] for r in common]
    return scores, dropped, attempted


def draw(ax, scores, dataset, sigmas, *, reference=True):
    nm = len(METHODS)
    gw = 0.78
    bw = gw / nm * 0.86
    whisk, fliers = [], []
    for j, (m, (label, color)) in enumerate(METHODS.items()):
        off = (j - (nm - 1) / 2) * (gw / nm)
        data, pos = [], []
        for i, s in enumerate(sigmas):
            v = scores.get((m, dataset, s), [])
            if v:
                data.append(v)
                pos.append(i + off)
        if not data:
            continue
        bp = ax.boxplot(data, positions=pos, widths=bw, patch_artist=True, showfliers=True,
                        medianprops=dict(color=SURFACE, linewidth=1.6),
                        whiskerprops=dict(color=color, linewidth=1.2),
                        capprops=dict(color=color, linewidth=1.2),
                        flierprops=dict(marker="o", markersize=3, markerfacecolor=color,
                                        markeredgecolor="none", alpha=0.75))
        for b in bp["boxes"]:
            b.set_facecolor(color); b.set_edgecolor(color)
            b.set_linewidth(1.2); b.set_alpha(0.85)
        for w in bp["whiskers"]:
            whisk += list(w.get_ydata())
        for f in bp["fliers"]:
            fliers += list(f.get_ydata())

    if whisk:
        lo, hi = min(whisk), max(whisk)
        pad = 0.08 * (hi - lo) or 1.0
        ax.set_ylim(lo - pad, hi + pad)

    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels([f"{s:g}" for s in sigmas])
    ax.set_xlim(-0.6, len(sigmas) - 0.4)
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)

    if reference:
        _add_reference(ax, dataset, xspan=(0, len(sigmas) - 1))

    lo, hi = ax.get_ylim()
    beyond = [v for v in fliers if not lo <= v <= hi]
    if beyond:
        top = max(beyond)
        ax.annotate(f"{len(beyond)} point{'s' if len(beyond) > 1 else ''} beyond axis "
                    f"(max {top:,.0f})" if top >= 1000 else
                    f"{len(beyond)} beyond axis (max {top:.2f})",
                    xy=(0.99, 0.99), xycoords="axes fraction", ha="right", va="top",
                    color=MUTED, fontsize=7.5,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))


def legend_handles():
    return [Line2D([0], [0], marker="s", markersize=9, linestyle="none",
                   markerfacecolor=c, markeredgecolor="none", label=l)
            for l, c in METHODS.values()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["lalonde_cps", "lalonde_psid"])
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    scores, dropped, attempted = load()
    n_reps = max(attempted.values(), default=0)

    for ds in args.datasets:
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        draw(ax, scores, ds, SIGMAS)
        ax.set_title(f"{DATASETS[ds]}   ({n_reps} draws, covariates contaminated)",
                     color=INK, fontsize=13, fontweight="bold", loc="left", pad=12)
        ax.set_xlabel(r"noise level  $\sigma$",
                      color=MUTED, fontsize=10)
        ax.set_ylabel("root-PEHE  (lower is better)", color=MUTED, fontsize=10)
        ax.legend(handles=legend_handles(), frameon=False, fontsize=9, labelcolor=MUTED,
                  loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, columnspacing=2.0)
        fig.tight_layout()
        out = FIGURES_DIR / f"contamination_{ds}{args.out_suffix}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")

    # --- 2x2 contact sheet, once all four datasets are present ------------
    if len(args.datasets) == len(DATASETS):
        panels = [d for d in DATASETS if d in args.datasets]  # canonical order
        fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4))
        for ax, ds in zip(axes.ravel(), panels):
            draw(ax, scores, ds, SIGMAS)
            ax.set_title(DATASETS[ds], color=INK, fontsize=12,
                         fontweight="bold", loc="left", pad=8)
        for ax in axes[-1]:
            ax.set_xlabel(r"noise level  $\sigma$", color=MUTED, fontsize=10)
        for ax in axes[:, 0]:
            ax.set_ylabel("root-PEHE  (lower is better)", color=MUTED, fontsize=10)
        fig.legend(handles=legend_handles(), frameon=False, fontsize=10, labelcolor=MUTED,
                   loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.005), columnspacing=2.5)
        fig.suptitle(f"root-PEHE vs covariate contamination   ({n_reps} draws per cell)",
                     color=INK, fontsize=13, fontweight="bold", x=0.055, ha="left", y=1.005)
        fig.tight_layout(rect=(0, 0.045, 1, 0.985))
        out = FIGURES_DIR / f"contamination_all_datasets{args.out_suffix}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")

    lines = [f"# PEHE vs covariate contamination ({n_reps} draws, median [Q1, Q3])", ""]
    for ds in args.datasets:
        lines += [f"## {DATASETS[ds]}", "",
                  "| Method | " + " | ".join(f"s={s:g} (rel {reliability(s):.3f})" for s in SIGMAS) + " |",
                  "|---" * (len(SIGMAS) + 1) + "|"]
        for m, (label, _) in METHODS.items():
            cells = []
            for s in SIGMAS:
                v = scores.get((m, ds, s), [])
                if not v:
                    cells.append("--"); continue
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                f = (lambda x: f"{x:,.0f}") if med >= 1000 else (lambda x: f"{x:.3f}")
                cells.append(f"{f(med)} [{f(q1)}, {f(q3)}]")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    out = RESULTS_DIR / f"summary_contamination{args.out_suffix}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
