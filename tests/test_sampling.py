import numpy as np
import pytest

from cate_benchmark.data import subsample_indices


def test_draws_are_deterministic_per_rep():
    np.testing.assert_array_equal(subsample_indices(1000, 100, 7), subsample_indices(1000, 100, 7))
    assert not np.array_equal(subsample_indices(1000, 100, 7), subsample_indices(1000, 100, 8))


def test_draws_are_nested_in_n():
    small, large = subsample_indices(1000, 100, 3), subsample_indices(1000, 400, 3)
    np.testing.assert_array_equal(small, large[:100])


def test_full_pool_draw_is_a_permutation():
    idx = subsample_indices(672, 672, 0)
    assert sorted(idx) == list(range(672))


def test_rejects_n_larger_than_pool():
    with pytest.raises(ValueError):
        subsample_indices(100, 101, 0)
