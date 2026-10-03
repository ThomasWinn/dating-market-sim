import numpy as np
import pytest

from sim.metrics import by_decile, gini


def test_gini_all_equal_is_zero():
    assert gini(np.full(10, 3.0)) == pytest.approx(0)


def test_gini_one_has_everything():
    x = np.zeros(10)
    x[3] = 5
    assert gini(x) == pytest.approx(9 / 10)


def test_gini_nobody_has_anything_is_zero():
    assert gini(np.zeros(5)) == 0


def test_gini_known_value():
    # Mean absolute difference over all ordered pairs / (2 * mean): for [1, 2, 3] = (8/9) / 4
    assert gini(np.array([1, 2, 3])) == pytest.approx(2 / 9)


def test_by_decile_averages_within_each_tenth():
    key = np.arange(100)
    values = np.repeat(np.arange(10.0), 10)  # decile d has value d
    np.testing.assert_allclose(by_decile(values, key), np.arange(10.0))


def test_by_decile_sorts_by_key_not_position():
    key = np.arange(100)[::-1]  # highest key first
    values = np.arange(100.0)
    assert by_decile(values, key)[0] == pytest.approx(np.arange(90, 100).mean())
