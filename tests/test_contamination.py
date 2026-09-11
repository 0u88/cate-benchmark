import numpy as np
import pytest

from cate_benchmark.contamination import Contaminator, resample_probability


@pytest.fixture(scope="module")
def pool():
    rng = np.random.default_rng(0)
    n = 40_000
    return np.column_stack([
        rng.normal(5.0, 2.0, n),           # numeric
        (rng.random(n) < 0.08).astype(float),  # rare binary, as in Lalonde's `black`
        (rng.random(n) < 0.5).astype(float),   # balanced binary
    ])


@pytest.mark.parametrize("sigma", [0.1, 1.0, 10.0])
def test_both_mechanisms_hit_the_target_correlation(pool, sigma):
    """corr(contaminated, clean) = 1/sqrt(1+sigma^2) for every column type."""
    con = Contaminator(pool)
    noisy = con.apply(pool, sigma, con.draw(pool.shape, [0, 0]))
    target = 1 / np.sqrt(1 + sigma**2)
    for j in range(pool.shape[1]):
        corr = np.corrcoef(pool[:, j], noisy[:, j])[0, 1]
        assert corr == pytest.approx(target, abs=0.02), f"column {j}"


def test_binary_columns_stay_binary(pool):
    con = Contaminator(pool)
    noisy = con.apply(pool, 1.0, con.draw(pool.shape, [0, 0]))
    assert set(np.unique(noisy[:, 1])) <= {0.0, 1.0}


def test_contamination_is_nested_in_sigma(pool):
    """The same noise draws are reused across levels: numeric noise scales with
    sigma, and cells resampled at a lower level are resampled at higher ones."""
    con = Contaminator(pool)
    noise = con.draw(pool.shape, [3, 0])
    low, high = con.apply(pool, 1.0, noise), con.apply(pool, 10.0, noise)
    np.testing.assert_allclose(high[:, 0] - pool[:, 0], 10 * (low[:, 0] - pool[:, 0]), rtol=1e-3, atol=1e-2)
    changed_low = low[:, 2] != pool[:, 2]
    assert np.all(high[changed_low, 2] == low[changed_low, 2])


def test_resample_probability_matches_numeric_reliability():
    for sigma in (0.01, 0.1, 1.0, 10.0):
        assert 1 - resample_probability(sigma) == pytest.approx(1 / np.sqrt(1 + sigma**2))
