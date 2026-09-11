"""The three CATE estimators, behind one interface.

    estimate = make_estimator(method, device)
    tau_hat = estimate(X_train, t_train, y_train, X_test)

Each estimator is used with its default settings. Model libraries are imported
lazily, because TabPFN/Do-PFN and CausalPFN need different Python environments.
"""
from __future__ import annotations

import sys
from collections.abc import Callable

import numpy as np

from . import config
from .data import working_directory

Estimator = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def tabpfn_t_learner(device: str) -> Estimator:
    """TabPFN v3 as a T-learner: one regressor per arm, tau_hat = mu1_hat - mu0_hat.

    An arm with too few rows makes TabPFN raise; the runner records such draws
    as failed rather than dropping them.
    """
    from tabpfn import TabPFNRegressor

    def estimate(X, t, y, X_test):
        arms = []
        for arm in (1, 0):
            model = TabPFNRegressor(device=device, random_state=0)
            model.fit(X[t == arm].astype(np.float32), y[t == arm].astype(np.float32))
            arms.append(np.asarray(model.predict(X_test.astype(np.float32))).ravel())
        return arms[0] - arms[1]

    return estimate


def causalpfn(device: str) -> Estimator:
    """CausalPFN, which estimates the CATE directly."""
    from causalpfn import CATEEstimator

    def estimate(X, t, y, X_test):
        model = CATEEstimator(device=device)
        model.fit(X, t, y)
        return np.asarray(model.estimate_cate(X=X_test)).ravel()

    return estimate


def dopfn(device: str) -> Estimator:
    """Do-PFN, which predicts the interventional outcome p(y | do(t), x).

    Upstream constraints: the treatment must be column 0, the model opens
    artifacts/*.pkl by relative path, and it runs on CPU only (an in-place
    Half/float dtype bug in model/layer.py breaks the CUDA forward pass).
    `predict_cate` mutates its input, so it always gets a copy.
    """
    del device  # CPU only, see above
    repo = str(config.DOPFN_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    with working_directory(repo):
        from scripts.transformer_prediction_interface.base import DoPFNRegressor

    def estimate(X, t, y, X_test):
        with working_directory(repo):
            model = DoPFNRegressor()
            model.fit(np.column_stack([t, X]).astype(np.float32), y.astype(np.float32))
            # Column 0 is a placeholder that predict_cate overwrites with 1, then 0.
            query = np.column_stack([np.zeros(len(X_test), np.float32), X_test])
            return np.asarray(model.predict_cate(query.astype(np.float32).copy())).ravel()

    return estimate


FACTORIES = {"tabpfn_t": tabpfn_t_learner, "causalpfn": causalpfn, "dopfn": dopfn}


def default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def make_estimator(method: str, device: str | None = None) -> Estimator:
    if method not in FACTORIES:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(FACTORIES)}")
    return FACTORIES[method](device or default_device())
