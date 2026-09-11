"""Shared constants and helpers for the CATE benchmark.

Deliberately dependency-light (numpy only) so it imports cleanly in every venv.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# Upstream model repos (not vendored). Default to clones under third_party/;
# override with the CAUSALPFN_REPO / DOPFN_REPO environment variables.
CAUSALPFN_REPO = Path(os.environ.get("CAUSALPFN_REPO", ROOT / "third_party" / "CausalPFN"))
DOPFN_REPO = Path(os.environ.get("DOPFN_REPO", ROOT / "third_party" / "Do-PFN"))

# Training subsample sizes (x-axis of every plot). Overridable so a second sweep
# can run at different sizes without disturbing the main one, e.g.
#   CATE_TRAIN_SIZES=2000,4000,6000,8000,10000 CATE_FIG_SUFFIX=_large python plot_boxplot.py
# Results merge by (dataset, n_train), so extra sizes add rows rather than
# overwrite; the plotting scripts only render the sizes it was asked for.
TRAIN_SIZES = [int(s) for s in os.environ.get("CATE_TRAIN_SIZES", "100,200,300,400,500").split(",")]

# Appended to figure filenames so a second sweep writes its own images.
FIG_SUFFIX = os.environ.get("CATE_FIG_SUFFIX", "")

# One realization per dataset, per the agreed design.
REALIZATION = 0

# Seed for the training subsample permutation. Fixed so all three methods
# receive byte-identical training data.
SUBSAMPLE_SEED = 0

# Keys are filesystem-safe slugs; values are the display names used in figures.
DATASETS = {
    "ihdp": "IHDP",
    "acic2016": "ACIC 2016",
    "lalonde_cps": "Lalonde CPS",
    "lalonde_psid": "Lalonde PSID",
}

METHODS = {
    "tabpfn_t": "TabPFN v3 (T-learner)",
    "tabpfn_s": "TabPFN v3 (S-learner)",
    "causalpfn": "CausalPFN",
    "dopfn": "Do-PFN",
}


def pehe(cate_true: np.ndarray, cate_pred: np.ndarray) -> float:
    """PEHE exactly as CausalPFN defines it (src/causalpfn/evaluation.py).

    Note this is the *square root* of the mean squared CATE error -- what most
    papers write as sqrt(PEHE). We keep CausalPFN's naming so the numbers are
    directly comparable to their published tables.
    """
    cate_true = np.asarray(cate_true, dtype=np.float64).ravel()
    cate_pred = np.asarray(cate_pred, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean((cate_true - cate_pred) ** 2)))


def npz_path(dataset: str, n_train: int) -> Path:
    return DATA_DIR / f"{dataset}_n{n_train}.npz"


def load_split(dataset: str, n_train: int) -> dict[str, np.ndarray]:
    """Load one prepared (dataset, train-size) split."""
    with np.load(npz_path(dataset, n_train)) as f:
        return {k: f[k] for k in f.files}


def save_results(method: str, rows: list[dict]) -> Path:
    """Write one method's results, merging with anything already on disk.

    Merging on (dataset, n_train) makes every runner resumable: rerunning after
    a crash keeps the rows that already succeeded.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{method}.json"
    existing = {}
    if out.exists():
        for r in json.loads(out.read_text(encoding="utf-8")):
            existing[(r["dataset"], r["n_train"])] = r
    for r in rows:
        existing[(r["dataset"], r["n_train"])] = r
    merged = sorted(
        existing.values(), key=lambda r: (list(DATASETS).index(r["dataset"]), r["n_train"])
    )
    out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return out


def stub_sklift() -> None:
    """CausalPFN's benchmarks/__init__.py imports scikit-uplift for the Criteo
    and Hillstrom datasets, which we do not use and is not installed. Register a
    permissive stub so the four loaders we *do* need are importable."""
    import sys
    import types

    for name in ("sklift", "sklift.datasets"):
        if name in sys.modules:
            continue
        m = types.ModuleType(name)
        m.__path__ = []
        m.__getattr__ = lambda attr: None
        sys.modules[name] = m


def load_causalpfn_dataset(dataset: str):
    """Return CausalPFN's CATE_Dataset for `dataset` at realization REALIZATION.

    Must chdir into the CausalPFN repo: the IHDP and RealCause loaders open
    their vendored data files by relative path.
    """
    import sys

    stub_sklift()
    repo = str(CAUSALPFN_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    cwd = os.getcwd()
    os.chdir(repo)
    try:
        if dataset == "ihdp":
            from benchmarks.ihdp import IHDPDataset

            catalog = IHDPDataset()
        elif dataset == "acic2016":
            from benchmarks.acic2016 import ACIC2016Dataset

            catalog = ACIC2016Dataset()
        elif dataset == "lalonde_cps":
            from benchmarks.realcause import RealCauseLalondeCPSDataset

            catalog = RealCauseLalondeCPSDataset()
        elif dataset == "lalonde_psid":
            from benchmarks.realcause import RealCauseLalondePSIDDataset

            catalog = RealCauseLalondePSIDDataset()
        else:
            raise ValueError(f"unknown dataset {dataset!r}")
        cate_dset, _ate_dset = catalog[REALIZATION]
        return cate_dset
    finally:
        os.chdir(cwd)
