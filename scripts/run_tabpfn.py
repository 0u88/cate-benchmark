"""TabPFN v3 as a CATE estimator, via either meta-learner.

S-learner: one regressor on [t, X] -> y, tau_hat(x) = f(1, x) - f(0, x).
T-learner: one regressor per arm, tau_hat(x) = mu1_hat(x) - mu0_hat(x).

Both are reported because the wrapper, not the base model, dominates the result
on the two Lalonde datasets. There treatment is rare (1.1% / 5.9%), so the
S-learner sees a near-constant t column, downweights it, and tau_hat collapses
toward 0 -- its PEHE lands almost exactly on the "predict zero" baseline. The
T-learner avoids this by construction and does 25-40% better at every size,
even with only 2 treated rows.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common import DATASETS, TRAIN_SIZES, load_split, pehe, save_results

SEED = 0


def _fit_predict(X_fit, y_fit, X_query, device: str) -> np.ndarray:
    from tabpfn import TabPFNRegressor

    model = TabPFNRegressor(device=device, random_state=SEED)
    model.fit(X_fit.astype(np.float32), y_fit.astype(np.float32))
    return np.asarray(model.predict(X_query.astype(np.float32))).ravel()


def s_learner_cate(X_train, t_train, y_train, X_test, device: str) -> np.ndarray:
    # Treatment first, matching the column convention used for Do-PFN.
    Xt_train = np.column_stack([t_train, X_train])
    ones = np.ones((len(X_test), 1), dtype=np.float32)
    from tabpfn import TabPFNRegressor

    model = TabPFNRegressor(device=device, random_state=SEED)
    model.fit(Xt_train.astype(np.float32), y_train.astype(np.float32))
    mu1 = model.predict(np.column_stack([ones, X_test]).astype(np.float32))
    mu0 = model.predict(np.column_stack([0 * ones, X_test]).astype(np.float32))
    return np.asarray(mu1).ravel() - np.asarray(mu0).ravel()


def t_learner_cate(X_train, t_train, y_train, X_test, device: str) -> np.ndarray:
    arms = []
    for arm in (1, 0):
        mask = t_train == arm
        if mask.sum() < 2:
            raise ValueError(f"only {int(mask.sum())} rows in arm t={arm}; cannot fit")
        arms.append(_fit_predict(X_train[mask], y_train[mask], X_test, device))
    return arms[0] - arms[1]


LEARNERS = {"s": ("tabpfn_s", s_learner_cate), "t": ("tabpfn_t", t_learner_cate)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--learner", choices=list(LEARNERS), default="s")
    args = ap.parse_args()
    method, cate_fn = LEARNERS[args.learner]

    rows = []
    for dataset in args.datasets:
        for n in TRAIN_SIZES:
            d = load_split(dataset, n)
            t0 = time.time()
            cate_pred = cate_fn(
                d["X_train"], d["t_train"], d["y_train"], d["X_test"], args.device
            )
            elapsed = time.time() - t0
            score = pehe(d["true_cate"], cate_pred)
            rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "n_train": n,
                    "pehe": score,
                    "seconds": elapsed,
                    "cate_pred_mean": float(np.mean(cate_pred)),
                    "cate_pred_sd": float(np.std(cate_pred)),
                }
            )
            print(
                f"{DATASETS[dataset]:13s} n={n:4d}  PEHE={score:12.4f}  "
                f"({elapsed:6.1f}s, pred mean={np.mean(cate_pred):.4g})",
                flush=True,
            )
            save_results(method, rows)


if __name__ == "__main__":
    main()
