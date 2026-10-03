"""Market metrics."""

import numpy as np


def gini(x: np.ndarray) -> float:
    """0 = everyone has the same amount; (n - 1) / n = one person has everything."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def by_decile(values: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Mean of `values` within each tenth of people sorted by `key` (lowest key first)."""
    order = np.argsort(key, kind="stable")
    return np.array([values[chunk].mean() for chunk in np.array_split(order, 10)])
