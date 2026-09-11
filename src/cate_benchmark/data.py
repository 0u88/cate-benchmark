"""Loading the benchmarks and drawing training subsamples.

The four datasets are read through CausalPFN's own benchmark loaders, so the
test sets and ground-truth CATE are exactly the ones CausalPFN publishes
results on (realization 0).
"""
from __future__ import annotations

import os
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from . import config


@dataclass(frozen=True)
class Pool:
    """A dataset's full training pool and its fixed test set, as float32 arrays."""

    X: np.ndarray
    t: np.ndarray
    y: np.ndarray
    X_test: np.ndarray
    true_cate: np.ndarray


@contextmanager
def working_directory(path):
    """Temporarily chdir. Upstream loaders and models open files by relative path."""
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _stub_sklift() -> None:
    """CausalPFN's benchmarks/__init__.py imports scikit-uplift for datasets we do
    not use. Register a permissive stub so the four loaders we need import."""
    for name in ("sklift", "sklift.datasets"):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = []
            module.__getattr__ = lambda attr: None
            sys.modules[name] = module


def load_causalpfn_dataset(dataset: str):
    """Return CausalPFN's CATE dataset object for `dataset` at REALIZATION."""
    _stub_sklift()
    repo = str(config.CAUSALPFN_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    with working_directory(repo):
        if dataset == "ihdp":
            from benchmarks.ihdp import IHDPDataset as Catalog
        elif dataset == "acic2016":
            from benchmarks.acic2016 import ACIC2016Dataset as Catalog
        elif dataset == "lalonde_cps":
            from benchmarks.realcause import RealCauseLalondeCPSDataset as Catalog
        elif dataset == "lalonde_psid":
            from benchmarks.realcause import RealCauseLalondePSIDDataset as Catalog
        else:
            raise ValueError(f"unknown dataset {dataset!r}")
        cate_dataset, _ate_dataset = Catalog()[config.REALIZATION]
    return cate_dataset


def load_pool(dataset: str) -> Pool:
    c = load_causalpfn_dataset(dataset)
    return Pool(
        X=np.asarray(c.X_train, dtype=np.float32),
        t=np.asarray(c.t_train, dtype=np.float32).ravel(),
        y=np.asarray(c.y_train, dtype=np.float32).ravel(),
        X_test=np.asarray(c.X_test, dtype=np.float32),
        true_cate=np.asarray(c.true_cate, dtype=np.float32).ravel(),
    )


def subsample_indices(pool_size: int, n: int, rep: int) -> np.ndarray:
    """Training rows for draw `rep`.

    `default_rng(rep).permutation(pool)[:n]` is nested in n (the n=100 draw is a
    prefix of the n=200 draw) and gives bit-identical permutations under numpy
    1.26 and 2.x, so methods running in different environments see the same rows.
    """
    if n > pool_size:
        raise ValueError(f"requested n={n} from a pool of {pool_size}")
    return np.random.default_rng(rep).permutation(pool_size)[:n]


def subsample(pool: Pool, n: int, rep: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = subsample_indices(len(pool.X), n, rep)
    return pool.X[idx], pool.t[idx], pool.y[idx]


def dataset_stats(pool: Pool) -> dict:
    """Summary statistics recorded in results/datasets.json.

    The two reference levels are stored here so that figures and tables can be
    rebuilt from the committed results without loading the datasets.
    """
    tau = pool.true_cate.astype(np.float64)
    return {
        "n_pool": int(len(pool.X)),
        "n_test": int(len(pool.X_test)),
        "n_covariates": int(pool.X.shape[1]),
        "treated_fraction": float(pool.t.mean()),
        "tau_mean": float(tau.mean()),
        "tau_sd": float(tau.std()),
        "y_sd": float(pool.y.astype(np.float64).std()),
        "predict_zero": float(np.sqrt(np.mean(tau**2))),
        "oracle": float(tau.std()),
    }
