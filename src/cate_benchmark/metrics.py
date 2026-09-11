"""Error metrics."""
from __future__ import annotations

import numpy as np


def pehe(cate_true, cate_pred) -> float:
    """Root-PEHE, exactly as CausalPFN defines it (src/causalpfn/evaluation.py).

    This is the square root of the mean squared CATE error, what most papers
    write as sqrt(PEHE). CausalPFN's naming is kept so the numbers are directly
    comparable to its published tables.
    """
    cate_true = np.asarray(cate_true, dtype=np.float64).ravel()
    cate_pred = np.asarray(cate_pred, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean((cate_true - cate_pred) ** 2)))


def reference_levels(cate_true) -> tuple[float, float]:
    """The two model-free reference errors for a test set.

    predict-zero  sqrt(mean(tau^2)): always predict tau_hat = 0. A method at
                  this level has extracted no signal.
    oracle        sd(tau): always predict the true mean effect. A method below
                  this level has captured genuine heterogeneity.
    """
    tau = np.asarray(cate_true, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean(tau**2))), float(tau.std())


def decompose(pehe_value: float, pred_mean: float, tau_mean: float) -> tuple[float, float]:
    """Split root-PEHE into the error in the average effect and everything else.

    Since PEHE^2 = bias^2 + residual^2 with bias = mean(tau_hat) - mean(tau).
    """
    bias = pred_mean - tau_mean
    residual = float(np.sqrt(max(pehe_value**2 - bias**2, 0.0)))
    return bias, residual
