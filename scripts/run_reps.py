"""Run one method across every (dataset, size, rep) of the repeated experiment.

Only the *training* draw varies: realization stays at 0, so the test set and the
true CATE are byte-identical across every rep. The spread in the resulting
boxplots is training-sample variability alone.

Rep r draws `np.random.default_rng(r).permutation(pool)[:n]`, which is nested in
n and reproducible across venvs -- PCG64 gives byte-identical permutations under
numpy 1.26 and 2.4, verified, so the two venvs see the same rows without any
intermediate files. Rep 0 reproduces the single-shot run in results/.

Results go to results/reps/<method><CATE_FIG_SUFFIX>.json, keyed by
(dataset, n_train, rep). Reruns merge, so a crashed run resumes by re-invoking
the same command, and `--rep-start` extends a finished run with more draws.

Which environment, per method (see requirements/):
    tabpfn_t, dopfn   ->  requirements/tabpfn-dopfn.txt  (Python 3.12)
    causalpfn         ->  requirements/causalpfn.txt     (Python 3.10)

Concurrent sweeps MUST use different CATE_FIG_SUFFIX values, or they will race
on the same results file.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from common import (
    DATASETS,
    FIG_SUFFIX,
    RESULTS_DIR,
    TRAIN_SIZES,
    load_causalpfn_dataset,
    pehe,
)

REPS_RESULTS = RESULTS_DIR / "reps"


def load_pool(dataset: str) -> dict:
    """Full training pool plus the fixed test set, as float32 arrays."""
    c = load_causalpfn_dataset(dataset)
    return {
        "X": np.asarray(c.X_train, dtype=np.float32),
        "t": np.asarray(c.t_train, dtype=np.float32).ravel(),
        "y": np.asarray(c.y_train, dtype=np.float32).ravel(),
        "X_test": np.asarray(c.X_test, dtype=np.float32),
        "true_cate": np.asarray(c.true_cate, dtype=np.float32).ravel(),
    }


def subsample(pool: dict, n: int, rep: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.random.default_rng(rep).permutation(len(pool["X"]))[:n]
    return pool["X"][idx], pool["t"][idx], pool["y"][idx]


def results_path(method: str):
    return REPS_RESULTS / f"{method}{FIG_SUFFIX}.json"


def load_existing(method: str) -> dict:
    """Already-computed rows, keyed by (dataset, n_train, rep)."""
    out = results_path(method)
    if not out.exists():
        return {}
    return {
        (r["dataset"], r["n_train"], r["rep"]): r
        for r in json.loads(out.read_text(encoding="utf-8"))
    }


def save(method: str, rows: list[dict]) -> None:
    REPS_RESULTS.mkdir(parents=True, exist_ok=True)
    existing = load_existing(method)
    for r in rows:
        existing[(r["dataset"], r["n_train"], r["rep"])] = r
    merged = sorted(
        existing.values(),
        key=lambda r: (list(DATASETS).index(r["dataset"]), r["n_train"], r["rep"]),
    )
    results_path(method).write_text(json.dumps(merged, indent=2), encoding="utf-8")


def make_estimator(method: str, device: str):
    """Return fn(X_train, t, y, X_test) -> per-unit CATE."""
    if method == "tabpfn_t":
        from tabpfn import TabPFNRegressor

        def fn(X, t, y, X_te):
            arms = []
            for arm in (1, 0):
                m = TabPFNRegressor(device=device, random_state=0)
                m.fit(X[t == arm].astype(np.float32), y[t == arm].astype(np.float32))
                arms.append(np.asarray(m.predict(X_te.astype(np.float32))).ravel())
            return arms[0] - arms[1]

        return fn

    if method == "causalpfn":
        from causalpfn import CATEEstimator

        def fn(X, t, y, X_te):
            est = CATEEstimator(device=device)
            est.fit(X, t, y)
            return np.asarray(est.estimate_cate(X=X_te)).ravel()

        return fn

    if method == "dopfn":
        import os
        import sys

        from common import DOPFN_REPO

        sys.path.insert(0, str(DOPFN_REPO))
        os.chdir(DOPFN_REPO)  # artifacts/*.pkl open by relative path
        from scripts.transformer_prediction_interface.base import DoPFNRegressor

        def fn(X, t, y, X_te):
            m = DoPFNRegressor()
            m.fit(np.column_stack([t, X]).astype(np.float32), y.astype(np.float32))
            Xq = np.column_stack([np.zeros(len(X_te), np.float32), X_te]).astype(np.float32)
            return np.asarray(m.predict_cate(Xq.copy())).ravel()  # predict_cate mutates

        return fn

    raise ValueError(f"unknown method {method!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["tabpfn_t", "causalpfn", "dopfn"])
    ap.add_argument("--reps", type=int, default=20, help="exclusive upper bound on rep index")
    ap.add_argument("--rep-start", type=int, default=0, help="skip reps below this index")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true", help="recompute cells already on disk")
    args = ap.parse_args()

    if args.device is None:
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    reps = range(args.rep_start, args.reps)
    # Anything already on disk is skipped, so an interrupted run resumes by
    # re-invoking the exact same command.
    done_keys = set(load_existing(args.method)) if not args.force else set()
    # Load pools before make_estimator, which chdirs away for Do-PFN.
    pools = {ds: load_pool(ds) for ds in args.datasets}
    estimate = make_estimator(args.method, args.device)

    todo = [
        (ds, n, rep)
        for ds in args.datasets
        for n in TRAIN_SIZES
        for rep in reps
        if (ds, n, rep) not in done_keys
    ]
    skipped = len(args.datasets) * len(TRAIN_SIZES) * len(reps) - len(todo)
    if skipped:
        print(f"resuming: {skipped} cells already on disk, {len(todo)} to go", flush=True)

    t_start = time.time()
    total = len(todo)
    done = 0
    for ds in args.datasets:
        pool = pools[ds]
        for n in TRAIN_SIZES:
            rows, scores, failures = [], [], []
            for rep in [r for (d, nn, r) in todo if d == ds and nn == n]:
                X, t, y = subsample(pool, n, rep)
                t0 = time.time()
                row = {
                    "method": args.method,
                    "dataset": ds,
                    "n_train": n,
                    "rep": rep,
                    "n_treated": int(t.sum()),
                }
                try:
                    pred = estimate(X, t, y, pool["X_test"])
                except Exception as exc:
                    # A uniform draw from Lalonde CPS (1.1% treated) can contain
                    # zero treated rows, which no per-arm learner can fit. Record
                    # it rather than dropping it, so the plot can disclose how
                    # many draws were unestimable.
                    failures.append(rep)
                    row |= {"pehe": None, "failed": f"{type(exc).__name__}: {exc}"}
                else:
                    score = pehe(pool["true_cate"], pred)
                    scores.append(score)
                    row |= {
                        "pehe": score,
                        "cate_pred_mean": float(np.mean(pred)),
                        "cate_pred_sd": float(np.std(pred)),
                    }
                row["seconds"] = time.time() - t0
                rows.append(row)
                done += 1
            if not rows:
                continue
            save(args.method, rows)
            eta = (time.time() - t_start) / done * (total - done)
            summary = (
                f"median={np.median(scores):12,.4f} [{min(scores):,.4f}, {max(scores):,.4f}]"
                if scores
                else f"{'ALL FAILED':>34}"
            )
            note = f"  {len(failures)} failed" if failures else ""
            print(
                f"{DATASETS[ds]:13s} n={n:5d}  PEHE {summary}{note}   "
                f"({done}/{total}, ETA {eta / 60:.1f} min)",
                flush=True,
            )


if __name__ == "__main__":
    main()
