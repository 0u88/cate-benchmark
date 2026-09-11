"""Dump the shared train/test splits used by all three methods.

For each dataset we take CausalPFN's realization 0 verbatim -- its test set and
its true CATE are used exactly as published -- then draw a *nested* uniform
random subsample of the training rows for each size in TRAIN_SIZES. Nested means
the n=100 subsample is a prefix of the n=200 one, so movement along the x-axis
is added data rather than a fresh draw.

Writing these to disk once guarantees the three runners (which live in different
venvs, with different numpy versions) see byte-identical training data.
"""
from __future__ import annotations

import numpy as np

from common import (
    DATA_DIR,
    DATASETS,
    SUBSAMPLE_SEED,
    TRAIN_SIZES,
    load_causalpfn_dataset,
    npz_path,
)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        cate = load_causalpfn_dataset(dataset)
        X_train = np.asarray(cate.X_train, dtype=np.float32)
        t_train = np.asarray(cate.t_train, dtype=np.float32).ravel()
        y_train = np.asarray(cate.y_train, dtype=np.float32).ravel()
        X_test = np.asarray(cate.X_test, dtype=np.float32)
        true_cate = np.asarray(cate.true_cate, dtype=np.float32).ravel()

        perm = np.random.default_rng(SUBSAMPLE_SEED).permutation(len(X_train))
        print(
            f"{DATASETS[dataset]:13s} pool={len(X_train):6d} d={X_train.shape[1]:3d} "
            f"test={len(X_test):5d} treated_frac={t_train.mean():.4f}"
        )
        for n in TRAIN_SIZES:
            if n > len(X_train):
                raise ValueError(f"{dataset}: only {len(X_train)} train rows, need {n}")
            idx = perm[:n]
            np.savez_compressed(
                npz_path(dataset, n),
                X_train=X_train[idx],
                t_train=t_train[idx],
                y_train=y_train[idx],
                X_test=X_test,
                true_cate=true_cate,
            )
            n_treated = int(t_train[idx].sum())
            print(f"    n={n:4d}  treated={n_treated:4d} ({100 * n_treated / n:5.1f}%)")


if __name__ == "__main__":
    main()
