"""CausalPFN as a CATE estimator.

Runs in the Python 3.10 environment from requirements/causalpfn.txt. Reads the same prepared .npz splits as the other two runners.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common import DATASETS, TRAIN_SIZES, load_split, pehe, save_results


def causalpfn_cate(X_train, t_train, y_train, X_test, device: str) -> np.ndarray:
    from causalpfn import CATEEstimator

    est = CATEEstimator(device=device)
    est.fit(X_train, t_train, y_train)
    return np.asarray(est.estimate_cate(X=X_test)).ravel()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    args = ap.parse_args()

    rows = []
    for dataset in args.datasets:
        for n in TRAIN_SIZES:
            d = load_split(dataset, n)
            t0 = time.time()
            cate_pred = causalpfn_cate(
                d["X_train"], d["t_train"], d["y_train"], d["X_test"], args.device
            )
            elapsed = time.time() - t0
            score = pehe(d["true_cate"], cate_pred)
            rows.append(
                {
                    "method": "causalpfn",
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
            save_results("causalpfn", rows)


if __name__ == "__main__":
    main()
