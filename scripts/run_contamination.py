"""Run one method across every (dataset, sigma, rep) of the contamination sweep.

Fixed training size per dataset; the swept variable is how badly the observed
confounders are measured. See contaminate.py for the noise model.

Results go to results/contamination/<method>.json, keyed by
(dataset, sigma, rep). Reruns skip cells already on disk.

Which environment, per method (see requirements/):
    tabpfn_t, dopfn   ->  requirements/tabpfn-dopfn.txt  (Python 3.12)
    causalpfn         ->  requirements/causalpfn.txt     (Python 3.10)
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from common import DATASETS, RESULTS_DIR, load_causalpfn_dataset, pehe
from contaminate import Contaminator
from run_reps import make_estimator

RESULTS = RESULTS_DIR / "contamination"

SIGMAS = [0.01, 0.1, 1.0, 10.0]

# Training size held fixed. Lalonde CPS sits at 2,000 because at n<=500 its 1.1%
# treated rate leaves a third of draws unfittable.
TRAIN_N = {"lalonde_cps": 2000, "lalonde_psid": 500, "ihdp": 500, "acic2016": 500}


def results_path(method: str):
    return RESULTS / f"{method}.json"


def load_existing(method: str) -> dict:
    p = results_path(method)
    if not p.exists():
        return {}
    return {
        (r["dataset"], r["sigma"], r["rep"]): r
        for r in json.loads(p.read_text(encoding="utf-8"))
    }


def save(method: str, rows: list[dict]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    existing = load_existing(method)
    for r in rows:
        existing[(r["dataset"], r["sigma"], r["rep"])] = r
    merged = sorted(
        existing.values(),
        key=lambda r: (list(DATASETS).index(r["dataset"]), r["sigma"], r["rep"]),
    )
    results_path(method).write_text(json.dumps(merged, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["tabpfn_t", "causalpfn", "dopfn"])
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--datasets", nargs="*", default=["lalonde_cps", "lalonde_psid"])
    ap.add_argument("--sigmas", nargs="*", type=float, default=SIGMAS)
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.device is None:
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    done = set() if args.force else set(load_existing(args.method))

    # Load pools before make_estimator, which chdirs away for Do-PFN.
    pools = {}
    for ds in args.datasets:
        c = load_causalpfn_dataset(ds)
        X = np.asarray(c.X_train, np.float32)
        pools[ds] = {
            "X": X,
            "t": np.asarray(c.t_train, np.float32).ravel(),
            "y": np.asarray(c.y_train, np.float32).ravel(),
            "X_test": np.asarray(c.X_test, np.float32),
            "true_cate": np.asarray(c.true_cate, np.float32).ravel(),
            "con": Contaminator(X),
        }
    estimate = make_estimator(args.method, args.device)

    todo = [
        (ds, s, r)
        for ds in args.datasets
        for s in args.sigmas
        for r in range(args.reps)
        if (ds, s, r) not in done
    ]
    skipped = len(args.datasets) * len(args.sigmas) * args.reps - len(todo)
    if skipped:
        print(f"resuming: {skipped} cells on disk, {len(todo)} to go", flush=True)

    t_start, total, count = time.time(), len(todo), 0
    for ds in args.datasets:
        pool, n = pools[ds], TRAIN_N[ds]
        con = pool["con"]
        for sigma in args.sigmas:
            rows, scores = [], []
            for rep in [r for (d, s, r) in todo if d == ds and s == sigma]:
                idx = np.random.default_rng(rep).permutation(len(pool["X"]))[:n]
                X, t, y = pool["X"][idx], pool["t"][idx], pool["y"][idx]
                X_te = pool["X_test"]
                # Nested in sigma: the same eps/u at every level, independent
                # between train and test.
                Xc = con.apply(X, sigma, con.draw(X.shape, [rep, 0]))
                Xc_te = con.apply(X_te, sigma, con.draw(X_te.shape, [rep, 1]))

                t0 = time.time()
                row = {
                    "method": args.method, "dataset": ds, "sigma": sigma, "rep": rep,
                    "n_train": n, "n_treated": int(t.sum()),
                }
                try:
                    pred = estimate(Xc, t, y, Xc_te)
                except Exception as exc:
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
                count += 1
            if not rows:
                continue
            save(args.method, rows)
            eta = (time.time() - t_start) / count * (total - count)
            summary = (
                f"median={np.median(scores):12,.1f} [{min(scores):,.1f}, {max(scores):,.1f}]"
                if scores else f"{'ALL FAILED':>34}"
            )
            nfail = len(rows) - len(scores)
            print(
                f"{DATASETS[ds]:13s} sigma={sigma:<6g} PEHE {summary}"
                f"{f'  {nfail} failed' if nfail else ''}   "
                f"({count}/{total}, ETA {eta/60:.1f} min)",
                flush=True,
            )


if __name__ == "__main__":
    main()
