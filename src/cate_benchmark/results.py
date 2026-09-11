"""Reading and writing per-draw results.

Each experiment stores one JSON file per method, results/<experiment>/<method>.json:
a list of rows, one per (dataset, n_train, sigma, rep) cell. Writes merge with
what is already on disk, which makes every run resumable.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from . import config
from .config import Experiment

Key = tuple[str, int, float | None, int]


def row_key(row: dict) -> Key:
    return (row["dataset"], row["n_train"], row.get("sigma"), row["rep"])


def _sort_key(row: dict):
    sigma = row.get("sigma")
    return (
        list(config.DATASETS).index(row["dataset"]),
        row["n_train"],
        -1.0 if sigma is None else sigma,
        row["rep"],
    )


def results_path(experiment: Experiment, method: str) -> Path:
    return experiment.results_dir / f"{method}.json"


def load_rows(experiment: Experiment, method: str) -> dict[Key, dict]:
    path = results_path(experiment, method)
    if not path.exists():
        return {}
    return {row_key(r): r for r in json.loads(path.read_text(encoding="utf-8"))}


def save_rows(experiment: Experiment, method: str, rows: list[dict]) -> Path:
    merged = load_rows(experiment, method)
    for r in rows:
        merged[row_key(r)] = r
    path = results_path(experiment, method)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(merged.values(), key=_sort_key)
    path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")
    return path


def common_scores(experiment: Experiment, methods=tuple(config.METHODS)):
    """Per-cell PEHEs over the draws that every method could fit.

    Returns (scores, dropped, attempted), keyed by (dataset, n_train, sigma):
      scores[cell][method] -> list of PEHEs, in rep order
      dropped[cell]        -> number of draws excluded
      attempted[cell]      -> number of draws that every method has a record for

    Restricting to common draws matters. A uniform draw from a sparsely treated
    dataset can contain too few treated units, and the methods disagree about
    what to do: CausalPFN and the TabPFN T-learner raise, while Do-PFN returns a
    number anyway. Summarizing each method over whatever it survived would
    compare different subsets of draws.
    """
    raw = {m: load_rows(experiment, m) for m in methods}
    cells = defaultdict(set)
    for rows in raw.values():
        for ds, n, sigma, rep in rows:
            cells[(ds, n, sigma)].add(rep)

    scores, dropped, attempted = {}, {}, {}
    for cell, reps in cells.items():
        computed = [r for r in sorted(reps) if all((*cell, r) in raw[m] for m in methods)]
        common = [
            r for r in computed if all(raw[m][(*cell, r)].get("pehe") is not None for m in methods)
        ]
        attempted[cell] = len(computed)
        dropped[cell] = len(computed) - len(common)
        scores[cell] = {m: [raw[m][(*cell, r)]["pehe"] for r in common] for m in methods}
    return scores, dropped, attempted


def load_dataset_stats() -> dict[str, dict]:
    if not config.DATASET_STATS.exists():
        raise FileNotFoundError(
            f"{config.DATASET_STATS} is missing; run `cate-bench datasets` to rebuild it"
        )
    return json.loads(config.DATASET_STATS.read_text(encoding="utf-8"))


def save_dataset_stats(stats: dict[str, dict]) -> None:
    merged = load_dataset_stats() if config.DATASET_STATS.exists() else {}
    merged.update(stats)
    ordered = {ds: merged[ds] for ds in config.DATASETS if ds in merged}
    config.DATASET_STATS.parent.mkdir(parents=True, exist_ok=True)
    config.DATASET_STATS.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")
