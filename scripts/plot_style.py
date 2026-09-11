"""Shared figure style and reference lines for the boxplot figures.

Colors come from a validated categorical palette (light mode). The two model-free
reference levels (predict-zero and oracle) are drawn by `_add_reference`.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import TRAIN_SIZES

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def _thousands(v, _pos):
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:g}"


def reference_levels(dataset: str) -> tuple[float, float]:
    """The two model-free reference PEHEs for a dataset.

    predict-zero  sqrt(mean(tau^2))  -- predict tau_hat = 0 for every unit.
                  A method sitting here has extracted nothing.
    oracle        sd(tau)            -- predict the true constant mean effect.
                  A method below here has captured real heterogeneity; between
                  the two it has only recovered the level.

    Both depend solely on the test set's true CATE, so they are constant across
    n and across methods. The oracle is the stricter reference and matters most
    where the CATE's mean dwarfs its spread (IHDP: mean 4.06, sd 0.79), because
    predict-zero flatters a method there.
    """
    import numpy as np

    from common import load_split

    tau = np.asarray(load_split(dataset, TRAIN_SIZES[0])["true_cate"], dtype=np.float64)
    return float(np.sqrt(np.mean(tau**2))), float(tau.std())


def _fmt(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.2f}"


def _add_reference(ax, dataset: str, xspan: tuple[float, float] | None = None) -> None:
    """Draw both reference levels, annotating any that fall off scale.

    Drawing a reference to scale is skipped when it would stretch the y-axis by
    more than 60% and flatten the series -- on IHDP predict-zero (4.13) sits far
    above every method (0.28-1.62). The oracle there (0.79) does fit, and is the
    informative one.

    `xspan` is where to anchor the two labels (left, right). It defaults to the
    training-size axis; the boxplots pass tick positions instead, since their
    x-axis is categorical.
    """
    x_left, x_right = xspan if xspan else (TRAIN_SIZES[0], TRAIN_SIZES[-1])
    ref0, oracle = reference_levels(dataset)
    lo, hi = ax.get_ylim()
    limit = hi + 0.6 * (hi - lo)  # computed before any expansion

    specs = [
        (oracle, "oracle: predict true mean", (0, (1, 2.5)), "right"),
        (ref0, "predict $\\hat\\tau$ = 0", (0, (5, 4)), "left"),
    ]
    fitting = [v for v, *_ in specs if v <= limit]
    if fitting:
        ax.set_ylim(lo, max(hi, max(fitting) + 0.06 * (hi - lo)))

    # Shade "level recovered but no heterogeneity". axhspan clips to the axes.
    if oracle <= limit:
        ax.axhspan(oracle, min(ref0, limit), color=MUTED, alpha=0.07, linewidth=0, zorder=0)

    offscale = []
    for value, name, dashes, side in specs:
        label = f"{name}  ({_fmt(value)})"
        if value > limit:
            offscale.append(label)
            continue
        ax.axhline(value, color=MUTED, linewidth=1.2, linestyle=dashes, zorder=1)
        at_left = side == "left"
        ax.annotate(
            label,
            xy=(x_left if at_left else x_right, value),
            xytext=(2 if at_left else -2, 4),
            textcoords="offset points",
            color=MUTED,
            fontsize=8,
            ha="left" if at_left else "right",
            va="bottom",
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5),
        )

    for i, label in enumerate(offscale):
        ax.annotate(
            f"{label}  ↑ off scale",
            xy=(0.99, 0.97 - 0.075 * i),
            xycoords="axes fraction",
            color=MUTED,
            fontsize=8,
            ha="right",
            va="top",
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5),
        )
