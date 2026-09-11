"""Figure 2: root-PEHE vs. covariate contamination, one panel per dataset.

x = noise level sigma, y = root-PEHE, one box per method at each level. The
reference lines depend only on the test set's true CATE, which contamination
leaves untouched, so they remain valid anchors here.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .. import config
from ..config import EXPERIMENTS
from ..results import common_scores
from .sample_size import draw_boxes, legend_handles
from .style import INK, MUTED


def plot(out: Path | None = None) -> Path:
    experiment = EXPERIMENTS["contamination"]
    scores, _dropped, attempted = common_scores(experiment)
    n_reps = max(attempted.values(), default=0)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4))
    for ax, (dataset, display) in zip(axes.ravel(), config.DATASETS.items()):
        n = experiment.train_sizes[dataset][0]
        columns = [(n, s) for s in experiment.sigmas]
        draw_boxes(ax, scores, dataset, columns, [f"{s:g}" for s in experiment.sigmas])
        ax.set_title(display, color=INK, fontsize=12, fontweight="bold", loc="left", pad=8)
    for ax in axes[-1]:
        ax.set_xlabel(r"noise level  $\sigma$", color=MUTED, fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("root-PEHE  (lower is better)", color=MUTED, fontsize=10)
    fig.legend(handles=legend_handles(), frameon=False, fontsize=10, labelcolor=MUTED,
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.005), columnspacing=2.5)
    fig.suptitle(f"root-PEHE vs covariate contamination   ({n_reps} draws per cell)",
                 color=INK, fontsize=13, fontweight="bold", x=0.055, ha="left", y=1.005)
    fig.tight_layout(rect=(0, 0.045, 1, 0.985))
    out = out or config.FIGURES_DIR / "fig2_contamination.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out
