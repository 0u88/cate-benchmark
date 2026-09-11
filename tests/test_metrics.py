import numpy as np
import pytest

from cate_benchmark.metrics import decompose, pehe, reference_levels


def test_pehe_is_root_mean_squared_error():
    assert pehe([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    assert pehe([0.0, 0.0], [3.0, 4.0]) == pytest.approx(np.sqrt(12.5))


def test_reference_levels_are_the_errors_of_the_two_constant_predictors():
    rng = np.random.default_rng(0)
    tau = rng.normal(2.0, 0.5, size=1000)
    predict_zero, oracle = reference_levels(tau)
    assert predict_zero == pytest.approx(pehe(tau, np.zeros_like(tau)))
    assert oracle == pytest.approx(pehe(tau, np.full_like(tau, tau.mean())))
    assert oracle <= predict_zero


def test_decompose_recovers_bias_and_residual():
    rng = np.random.default_rng(1)
    tau = rng.normal(size=500)
    tau_hat = tau + 0.3 + rng.normal(scale=0.2, size=500)
    bias, residual = decompose(pehe(tau, tau_hat), tau_hat.mean(), tau.mean())
    assert bias == pytest.approx(tau_hat.mean() - tau.mean())
    assert bias**2 + residual**2 == pytest.approx(pehe(tau, tau_hat) ** 2)
