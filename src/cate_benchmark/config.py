"""Paths, datasets, methods, and the registry of experiments.

Every experiment in the study is defined once, here. The runner, the plots and
the tables all read these definitions, so there is no per-run configuration to
keep in sync (the old scripts used environment variables for this).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Repository root. Results and figures live in the repository, not the package.
ROOT = Path(os.environ.get("CATE_BENCHMARK_ROOT", Path(__file__).resolve().parents[2]))
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
DATASET_STATS = RESULTS_DIR / "datasets.json"

# Upstream model repositories (not vendored). Default to clones under
# third_party/; override with the CAUSALPFN_REPO / DOPFN_REPO environment variables.
CAUSALPFN_REPO = Path(os.environ.get("CAUSALPFN_REPO", ROOT / "third_party" / "CausalPFN"))
DOPFN_REPO = Path(os.environ.get("DOPFN_REPO", ROOT / "third_party" / "Do-PFN"))

# One realization of each benchmark; only the training draw varies between reps.
REALIZATION = 0

# Keys are filesystem-safe slugs; values are display names.
DATASETS = {
    "ihdp": "IHDP",
    "acic2016": "ACIC 2016",
    "lalonde_cps": "Lalonde CPS",
    "lalonde_psid": "Lalonde PSID",
}

METHODS = {
    "tabpfn_t": "TabPFN v3 (T-learner)",
    "causalpfn": "CausalPFN",
    "dopfn": "Do-PFN",
}

# Size of each dataset's training pool (realization 0).
POOL_SIZES = {"ihdp": 672, "acic2016": 4321, "lalonde_cps": 14559, "lalonde_psid": 2407}

SMALL_SIZES = (100, 200, 300, 400, 500)
LARGE_SIZES = (2000, 4000, 6000, 8000, 10000)
SIGMAS = (0.01, 0.1, 1.0, 10.0)


@dataclass(frozen=True)
class Experiment:
    """One experimental sweep.

    Each cell is (dataset, n_train, sigma, rep). `sigmas` is empty for sweeps
    without covariate contamination.
    """

    name: str
    title: str
    train_sizes: dict[str, tuple[int, ...]]
    reps: int
    sigmas: tuple[float, ...] = field(default=())

    @property
    def datasets(self) -> tuple[str, ...]:
        return tuple(self.train_sizes)

    @property
    def results_dir(self) -> Path:
        return RESULTS_DIR / self.name

    @property
    def contaminated(self) -> bool:
        return bool(self.sigmas)


EXPERIMENTS = {
    e.name: e
    for e in (
        Experiment(
            name="full-pool",
            title="Experiment 1: baseline CATE estimation on the full training pool",
            train_sizes={ds: (n,) for ds, n in POOL_SIZES.items()},
            reps=1,
        ),
        Experiment(
            name="sample-size",
            title="Experiment 2: CATE vs. training-set size",
            train_sizes={ds: SMALL_SIZES for ds in DATASETS},
            reps=100,
        ),
        Experiment(
            name="sample-size-large",
            # Lalonde CPS is 1.06% treated: at n <= 500 most draws contain too
            # few treated units to fit, so its sweep uses larger n.
            title="Experiment 2 (Lalonde CPS): CATE vs. training-set size at large n",
            train_sizes={"lalonde_cps": LARGE_SIZES},
            reps=100,
        ),
        Experiment(
            name="contamination",
            title="Experiment 3: CATE vs. covariate contamination",
            train_sizes={ds: (2000,) if ds == "lalonde_cps" else (500,) for ds in DATASETS},
            reps=100,
            sigmas=SIGMAS,
        ),
    )
}
