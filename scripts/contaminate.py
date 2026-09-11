"""Corrupt the observed confounders by a controlled amount.

The experiment varies *measurement quality* of the covariates instead of the
training-set size. Both the training and the test covariates are corrupted --
the estimator never sees clean confounders -- while the treatment, the outcome
and the true CATE are left untouched, so the score still asks "did you get the
real effect right".

Two mechanisms, because one does not fit both column types:

  numeric      x + sigma * sd(x) * eps,  eps ~ N(0,1)
  binary/cat   flip (or resample from the column's empirical marginal)
               with probability p

Gaussian noise on a 0/1 column is wrong on two counts. Semantically, "married =
0.37" is not a measurement anyone could record; the textbook error model for a
discrete variable is misclassification. Technically it is worse: TabPFN infers
column types by distinct-value count, so *any* additive noise flips a binary
column from categorical to numerical. Measured on these datasets, sigma = 0.01
-- a 1% -of-a-SD perturbation, i.e. essentially no information loss -- already
drops IHDP from 19 inferred categorical columns to 0. The preprocessing pipeline
would change discontinuously at the very first sigma and confound the sweep.

The two mechanisms are put on a common scale through *reliability*, the
correlation between the observed and the true value:

  numeric    corr = 1 / sqrt(1 + sigma^2)
  resample   corr = 1 - p                    (exactly, see below)

so p = 1 - 1/sqrt(1+sigma^2) makes one sigma mean the same attenuation whatever
the column type:

  sigma  0.01 -> p 0.00005       sigma  1.0 -> p 0.293
  sigma  0.1  -> p 0.00496       sigma 10.0 -> p 0.901

Resampling from the column's own marginal is used rather than flipping. For a
binary column with prevalence q, flipping a fraction p gives
corr = (1-2p) * sqrt(q(1-q) / (r(1-r))) with r = q(1-p) + (1-q)p: because r is
pulled toward 0.5, rare categories lose far more reliability than the formula
suggests. Measured on Lalonde at sigma=1, `black` (q = 0.08) came out at 0.486
against a target of 0.707. Resampling instead leaves the marginal unchanged
(r = q), and then Cov = q(1-q)(1-p) with Var = q(1-q), i.e. corr = 1-p exactly,
for any prevalence and any number of levels.

Draws are *nested* in sigma: one eps and one uniform u per cell, reused across
sigmas. Larger sigma scales the same eps and flips a superset of the same cells,
so the sweep is paired and monotone rather than four unrelated perturbations.
"""
from __future__ import annotations

import numpy as np

# A column with at most this many distinct values is corrupted by resampling
# rather than by additive noise. Matches TabPFN's own inference on these data.
MAX_UNIQUE_FOR_CATEGORICAL = 2


def resample_probability(sigma: float) -> float:
    """Fraction of cells to redraw, matching a numeric column's reliability."""
    return 1.0 - 1.0 / np.sqrt(1.0 + sigma**2)


def column_kinds(X_pool: np.ndarray) -> np.ndarray:
    """True where a column should be treated as categorical."""
    return np.array(
        [len(np.unique(X_pool[:, j])) <= MAX_UNIQUE_FOR_CATEGORICAL for j in range(X_pool.shape[1])]
    )


def reference_sd(X_pool: np.ndarray) -> np.ndarray:
    """Per-column SD from the full training pool.

    Taken from the pool rather than the drawn subsample so that a given sigma
    means exactly the same perturbation in every repetition.
    """
    sd = X_pool.std(axis=0)
    sd[sd == 0] = 1.0  # a constant column has no scale; leave it unperturbed in effect
    return sd


class Contaminator:
    """Applies the same nested noise pattern at any sigma."""

    def __init__(self, X_pool: np.ndarray, categorical: np.ndarray | None = None):
        X_pool = np.asarray(X_pool, dtype=np.float64)
        self.sd = reference_sd(X_pool)
        self.categorical = column_kinds(X_pool) if categorical is None else categorical
        # The pool column itself is the empirical marginal to redraw from.
        self.pool = [
            np.sort(X_pool[:, j]) if self.categorical[j] else None
            for j in range(X_pool.shape[1])
        ]

    def draw(self, shape: tuple[int, int], seed) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """eps for numeric columns, u for the flip decision, and replacement draws."""
        rng = np.random.default_rng(seed)
        eps = rng.normal(size=shape)
        u = rng.random(size=shape)
        alt = rng.random(size=shape)  # picks the replacement level when a cell flips
        return eps, u, alt

    def apply(self, X: np.ndarray, sigma: float, noise) -> np.ndarray:
        """Return a corrupted copy of X at contamination level `sigma`."""
        eps, u, alt = noise
        out = np.asarray(X, dtype=np.float64).copy()
        p = resample_probability(sigma)
        for j in range(out.shape[1]):
            if self.categorical[j]:
                marginal = self.pool[j]
                if len(np.unique(marginal)) < 2:
                    continue
                hit = u[:, j] < p
                if not hit.any():
                    continue
                # Independent draw from the column's empirical marginal. It may
                # coincide with the current value -- that is what makes the
                # reliability exactly 1-p regardless of how rare a level is.
                pick = (alt[hit, j] * len(marginal)).astype(int).clip(0, len(marginal) - 1)
                out[hit, j] = marginal[pick]
            else:
                out[:, j] = out[:, j] + sigma * self.sd[j] * eps[:, j]
        return out.astype(np.float32)
