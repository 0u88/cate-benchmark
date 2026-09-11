"""Why Do-PFN fails, part 1: it cannot use more covariates than it was trained on.

Mechanism, read off the loaded Do-PFN v1 checkpoint:

    encoder = NanHandling -> VariableNumFeatures(85) -> ColumnMarker
              -> InputNormalization -> VariableNumFeatures(85) -> Linear(170 -> 192)

A table with d covariates (plus the treatment) is zero-padded to 85 columns and
embedded as a single token per row. Column j always lands in slot j of the
input Linear, and pre-training only populated slots 0-6 (the treatment plus at
most 6 covariates), so the weights for later slots never received gradient.
Covariates beyond the sixth are multiplied by effectively untrained weights.

Prediction: restricting a wide dataset to about 6 covariates should *improve*
Do-PFN, while it should hurt a model without this limit (TabPFN).

This diagnostic needs the models. It uses the n = 500 training draw of rep 0
and, for each covariate count d, three random covariate subsets (all
covariates when d is the full width). Results go to
results/diagnostics/dopfn_width.json.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .. import config
from ..data import load_pool, subsample
from ..estimators import make_estimator
from ..metrics import pehe

N_TRAIN = 500
SUBSETS_PER_D = 3
DIMS = (2, 4, 6, 8, 12, 16, 20, 25)
OUT = config.RESULTS_DIR / "diagnostics" / "dopfn_width.json"


def covariate_subsets(d_full: int, k: int) -> list[np.ndarray]:
    if k == d_full:
        return [np.arange(k)]
    rng = np.random.default_rng(0)
    return [np.sort(rng.choice(d_full, k, replace=False)) for _ in range(SUBSETS_PER_D)]


def run(datasets=("ihdp", "acic2016"), methods=("dopfn",), device=None) -> list[dict]:
    pools = {ds: load_pool(ds) for ds in datasets}
    estimators = {m: make_estimator(m, device) for m in methods}
    rows = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else []
    done = {(r["method"], r["dataset"], r["d"]) for r in rows}
    for ds in datasets:
        pool = pools[ds]
        X, t, y = subsample(pool, N_TRAIN, rep=0)
        d_full = X.shape[1]
        for k in sorted({k for k in DIMS if k <= d_full} | {d_full}):
            for method, estimate in estimators.items():
                if (method, ds, k) in done:
                    continue
                scores = [
                    pehe(pool.true_cate, estimate(X[:, cols], t, y, pool.X_test[:, cols]))
                    for cols in covariate_subsets(d_full, k)
                ]
                rows.append({"method": method, "dataset": ds, "d": k, "d_full": d_full,
                             "pehe_subsets": scores, "pehe_mean": float(np.mean(scores))})
                print(f"{config.DATASETS[ds]:10s} d={k:3d}  {config.METHODS[method]:22s} "
                      f"mean PEHE {np.mean(scores):.3f}", flush=True)
                OUT.parent.mkdir(parents=True, exist_ok=True)
                OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="cate-bench diagnose dopfn-width")
    ap.add_argument("--datasets", nargs="*", default=["ihdp", "acic2016"])
    ap.add_argument("--methods", nargs="*", default=["dopfn"], choices=["dopfn", "tabpfn_t"])
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    run(args.datasets, args.methods, args.device)
