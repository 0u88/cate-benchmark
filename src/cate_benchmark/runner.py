"""One resumable runner for every experiment.

For each cell (dataset, n_train, sigma, rep) the runner draws the training rows
for `rep`, optionally contaminates the covariates, fits the method, and scores
it against the true CATE of the fixed test set. Results are written after each
(dataset, n_train, sigma) group, so an interrupted run resumes by re-issuing the
same command.

Do not pipe the output (`| tail`) when running in the background: once the
parent shell exits, the process blocks forever writing to stdout.
"""
from __future__ import annotations

import time

import numpy as np

from . import config
from .config import Experiment
from .contamination import Contaminator
from .data import dataset_stats, load_pool, subsample
from .estimators import make_estimator
from .metrics import pehe
from .results import load_rows, save_dataset_stats, save_rows


def planned_cells(experiment: Experiment, datasets, reps: range):
    for ds in datasets:
        for n in experiment.train_sizes[ds]:
            for sigma in experiment.sigmas or (None,):
                for rep in reps:
                    yield ds, n, sigma, rep


def run(
    experiment: Experiment,
    method: str,
    *,
    reps: int | None = None,
    rep_start: int = 0,
    datasets=None,
    device: str | None = None,
    force: bool = False,
    pool_loader=load_pool,
    estimator_factory=make_estimator,
    log=print,
) -> int:
    """Run `method` over the experiment's cells. Returns the number of cells computed.

    `pool_loader` and `estimator_factory` are injectable for testing.
    """
    datasets = list(datasets or experiment.datasets)
    unknown = set(datasets) - set(experiment.datasets)
    if unknown:
        raise ValueError(f"{experiment.name} does not include {sorted(unknown)}")
    rep_range = range(rep_start, experiment.reps if reps is None else reps)

    done = set() if force else set(load_rows(experiment, method))
    todo = [c for c in planned_cells(experiment, datasets, rep_range) if c not in done]
    if not todo:
        log(f"{experiment.name}/{method}: nothing to do")
        return 0
    skipped = len(list(planned_cells(experiment, datasets, rep_range))) - len(todo)
    if skipped:
        log(f"resuming: {skipped} cells already on disk, {len(todo)} to go")

    pools = {ds: pool_loader(ds) for ds in datasets}
    if pool_loader is load_pool:
        save_dataset_stats({ds: dataset_stats(p) for ds, p in pools.items()})
    contaminators = (
        {ds: Contaminator(pools[ds].X) for ds in datasets} if experiment.contaminated else {}
    )
    estimate = estimator_factory(method, device)

    groups: dict[tuple, list[int]] = {}
    for ds, n, sigma, rep in todo:
        groups.setdefault((ds, n, sigma), []).append(rep)

    started, count = time.time(), 0
    for (ds, n, sigma), group_reps in groups.items():
        pool = pools[ds]
        rows, scores = [], []
        for rep in group_reps:
            X, t, y = subsample(pool, n, rep)
            X_test = pool.X_test
            row = {"method": method, "dataset": ds}
            if sigma is not None:
                # Nested in sigma: the same noise draws at every level,
                # independent between the training and the test covariates.
                con = contaminators[ds]
                X = con.apply(X, sigma, con.draw(X.shape, [rep, 0]))
                X_test = con.apply(X_test, sigma, con.draw(X_test.shape, [rep, 1]))
                row["sigma"] = sigma
            row |= {"n_train": n, "rep": rep, "n_treated": int(t.sum())}

            t0 = time.time()
            try:
                pred = estimate(X, t, y, X_test)
            except Exception as exc:  # recorded, so figures can disclose unfittable draws
                row |= {"pehe": None, "failed": f"{type(exc).__name__}: {exc}"}
            else:
                score = pehe(pool.true_cate, pred)
                scores.append(score)
                row |= {
                    "pehe": score,
                    "cate_pred_mean": float(np.mean(pred)),
                    "cate_pred_sd": float(np.std(pred)),
                }
            row["seconds"] = time.time() - t0
            rows.append(row)
            count += 1

        save_rows(experiment, method, rows)
        eta = (time.time() - started) / count * (len(todo) - count)
        level = f" sigma={sigma:<5g}" if sigma is not None else ""
        failed = len(rows) - len(scores)
        summary = f"median PEHE {np.median(scores):,.4f}" if scores else "all draws failed"
        log(
            f"{config.DATASETS[ds]:13s} n={n:<6d}{level} {summary}"
            f"{f'  ({failed} failed)' if failed else ''}  [{count}/{len(todo)}, ETA {eta / 60:.1f} min]"
        )
    return count
