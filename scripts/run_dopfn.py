"""Do-PFN as a CATE estimator.

Three hard requirements from the upstream implementation:
  1. The treatment must be column 0 -- `predict_cid` does `X[:, 0] = t`, and the
     pretraining prior always places the treatment key first.
  2. The process must run with cwd == the Do-PFN repo root, because
     `DoPFNRegressor.__init__` and `load_model` open `artifacts/*.pkl` by
     relative path.
  3. CPU only. `model/layer.py` has an in-place Half/float dtype bug that breaks
     the CUDA forward pass in this checkout.

Also note `predict_cate` mutates its argument in place, so we always hand it a
copy.

Caveat worth carrying into the write-up: Do-PFN was pretrained on 1-6 covariates
(checkpoint `max_num_features_in_training = 6`). IHDP has 25 and ACIC 2016 has
58, so those two datasets are well outside its training distribution.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from common import DATASETS, DOPFN_REPO, TRAIN_SIZES, load_split, pehe, save_results


def _import_dopfn():
    repo = str(DOPFN_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    os.chdir(repo)  # artifacts/*.pkl are opened by relative path
    from scripts.transformer_prediction_interface.base import DoPFNRegressor

    return DoPFNRegressor


def dopfn_cate(DoPFNRegressor, X_train, t_train, y_train, X_test) -> np.ndarray:
    Xt_train = np.column_stack([t_train, X_train]).astype(np.float32)
    # Column 0 is a placeholder; predict_cate overwrites it with 1 then 0.
    Xt_test = np.column_stack([np.zeros(len(X_test), np.float32), X_test]).astype(np.float32)

    model = DoPFNRegressor()
    model.fit(Xt_train, y_train.astype(np.float32))
    return np.asarray(model.predict_cate(Xt_test.copy())).ravel()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    args = ap.parse_args()

    # Load every split before chdir'ing away, so relative paths stay simple.
    splits = {(ds, n): load_split(ds, n) for ds in args.datasets for n in TRAIN_SIZES}
    DoPFNRegressor = _import_dopfn()

    rows = []
    for dataset in args.datasets:
        for n in TRAIN_SIZES:
            d = splits[(dataset, n)]
            t0 = time.time()
            cate_pred = dopfn_cate(
                DoPFNRegressor, d["X_train"], d["t_train"], d["y_train"], d["X_test"]
            )
            elapsed = time.time() - t0
            score = pehe(d["true_cate"], cate_pred)
            rows.append(
                {
                    "method": "dopfn",
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
            save_results("dopfn", rows)


if __name__ == "__main__":
    main()
