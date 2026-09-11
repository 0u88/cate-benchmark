"""Markdown tables built from the committed results (no models or data needed)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config
from .config import EXPERIMENTS
from .results import common_scores, load_dataset_stats, load_rows

TABLES_DIR = config.RESULTS_DIR / "tables"


def _fmt(x: float, large: bool) -> str:
    return f"{x:,.0f}" if large else f"{x:.3f}"


def full_pool_normalized() -> dict[str, dict[str, float]]:
    """Experiment 1: root-PEHE on the full training pool divided by the oracle
    error sd(tau). Values below 1 beat a constant predictor that knows the true
    average effect. Returns values[method][dataset]."""
    stats = load_dataset_stats()
    experiment = EXPERIMENTS["full-pool"]
    values = {}
    for method in config.METHODS:
        rows = load_rows(experiment, method)
        values[method] = {
            ds: rows[(ds, experiment.train_sizes[ds][0], None, 0)]["pehe"] / stats[ds]["oracle"]
            for ds in experiment.datasets
        }
    return values


def full_pool_table() -> str:
    values = full_pool_normalized()
    datasets = list(config.DATASETS)
    lines = [
        "# Experiment 1: oracle-normalized root-PEHE on the full training pool",
        "",
        "Each value is root-PEHE divided by sd(tau), the error of a constant predictor that",
        "knows the true average effect. Values below 1 beat that predictor. Best per dataset in bold.",
        "",
        "| Model | " + " | ".join(config.DATASETS[d] for d in datasets) + " |",
        "|---" * (len(datasets) + 1) + "|",
    ]
    best = {d: min(values[m][d] for m in config.METHODS) for d in datasets}
    for method, label in config.METHODS.items():
        cells = []
        for d in datasets:
            v = values[method][d]
            cells.append(f"**{v:.3f}**" if v == best[d] else f"{v:.3f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def sweep_table(experiment_name: str) -> str:
    """Median [Q1, Q3] root-PEHE per cell, over the draws every method could fit."""
    experiment = EXPERIMENTS[experiment_name]
    scores, dropped, _ = common_scores(experiment)
    axis = experiment.sigmas if experiment.contaminated else None
    lines = [f"# {experiment.title}", "", "Median [Q1, Q3] root-PEHE over training draws.", ""]
    for ds in experiment.datasets:
        n_values = experiment.train_sizes[ds]
        columns = [(n_values[0], s) for s in axis] if axis else [(n, None) for n in n_values]
        header = [f"σ={s:g}" for s in axis] if axis else [f"n={n:,}" for n in n_values]
        lines += [f"## {config.DATASETS[ds]}", ""]
        if axis:
            lines += [f"Training size n = {n_values[0]:,}.", ""]
        lines += ["| Method | " + " | ".join(header) + " |", "|---" * (len(columns) + 1) + "|"]
        for method, label in config.METHODS.items():
            cells = []
            for n, sigma in columns:
                v = scores.get((ds, n, sigma), {}).get(method, [])
                if not v:
                    cells.append("–")
                    continue
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                large = med >= 1000
                cells.append(f"{_fmt(med, large)} [{_fmt(q1, large)}, {_fmt(q3, large)}]")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        excluded = [
            (n, sigma, dropped[(ds, n, sigma)]) for n, sigma in columns if dropped.get((ds, n, sigma))
        ]
        if excluded:
            notes = ", ".join(f"n={n:,}: {k}" for n, _, k in excluded)
            lines += ["", f"Draws excluded because at least one method could not fit them: {notes}."]
        lines.append("")
    return "\n".join(lines)


def write_all(out_dir: Path = TABLES_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "full_pool.md": full_pool_table(),
        "sample_size.md": sweep_table("sample-size"),
        "sample_size_large.md": sweep_table("sample-size-large"),
        "contamination.md": sweep_table("contamination"),
    }
    paths = []
    for name, text in outputs.items():
        path = out_dir / name
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths
