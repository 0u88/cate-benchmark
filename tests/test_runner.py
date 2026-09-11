import numpy as np
import pytest

from cate_benchmark.config import Experiment
from cate_benchmark.data import Pool
from cate_benchmark.results import common_scores, load_rows
from cate_benchmark.runner import run


def fake_pool(_dataset):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3)).astype(np.float32)
    t = (rng.random(300) < 0.3).astype(np.float32)
    X_test = rng.normal(size=(50, 3)).astype(np.float32)
    tau = X_test[:, 0]
    return Pool(X=X, t=t, y=X[:, 0] * t, X_test=X_test, true_cate=tau)


def perfect_estimator(_method, _device):
    return lambda X, t, y, X_test: X_test[:, 0]


def failing_on_odd_reps(_method, _device):
    def estimate(X, t, y, X_test):
        if int(X[0, 0] * 1e6) % 2:  # deterministic per draw
            raise ValueError("unfittable")
        return X_test[:, 0] + 1.0
    return estimate


QUIET = dict(pool_loader=fake_pool, log=lambda *_: None)
SIZE = Experiment("toy", "toy", {"ihdp": (50, 100)}, reps=4)
NOISY = Experiment("toy-noise", "toy", {"ihdp": (100,)}, reps=3, sigmas=(0.1, 1.0))


def test_runs_every_cell_and_scores_it(tmp_results):
    assert run(SIZE, "dopfn", estimator_factory=perfect_estimator, **QUIET) == 8
    rows = load_rows(SIZE, "dopfn")
    assert set(rows) == {("ihdp", n, None, r) for n in (50, 100) for r in range(4)}
    assert all(r["pehe"] == pytest.approx(0.0) for r in rows.values())


def test_resumes_without_recomputing(tmp_results):
    run(SIZE, "dopfn", reps=2, estimator_factory=perfect_estimator, **QUIET)
    computed = run(SIZE, "dopfn", estimator_factory=perfect_estimator, **QUIET)
    assert computed == 4  # only reps 2 and 3 were left


def test_failed_draws_are_recorded_and_excluded_from_comparisons(tmp_results):
    run(SIZE, "dopfn", estimator_factory=perfect_estimator, **QUIET)
    run(SIZE, "causalpfn", estimator_factory=failing_on_odd_reps, **QUIET)
    failed = [r for r in load_rows(SIZE, "causalpfn").values() if r["pehe"] is None]
    assert failed and all("unfittable" in r["failed"] for r in failed)
    scores, dropped, attempted = common_scores(SIZE, methods=("dopfn", "causalpfn"))
    for cell in scores:
        assert len(scores[cell]["dopfn"]) == len(scores[cell]["causalpfn"]) == attempted[cell] - dropped[cell]


def test_contaminated_cells_carry_sigma_and_use_the_same_training_rows(tmp_results):
    seen = []

    def recorder(_method, _device):
        def estimate(X, t, y, X_test):
            seen.append(t.copy())
            return X_test[:, 0]
        return estimate

    run(NOISY, "dopfn", estimator_factory=recorder, **QUIET)
    rows = load_rows(NOISY, "dopfn")
    assert {k[2] for k in rows} == {0.1, 1.0}
    # Rep r uses the same training rows at every sigma: only the covariates change.
    by_rep = {}
    for (_, _, _sigma, rep), t in zip(sorted(rows, key=lambda k: (k[2], k[3])), seen):
        by_rep.setdefault(rep, []).append(t)
    assert all(np.array_equal(ts[0], ts[1]) for ts in by_rep.values())
