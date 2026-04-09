from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def to_numpy(values: Iterable[float]) -> np.ndarray:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("Expected at least one value.")
    return arr


def mean(values: Iterable[float]) -> float:
    arr = to_numpy(values)
    return float(np.mean(arr))


def variance(values: Iterable[float]) -> float:
    arr = to_numpy(values)
    return float(np.var(arr))


def minimum(values: Iterable[float]) -> float:
    arr = to_numpy(values)
    return float(np.min(arr))


def maximum(values: Iterable[float]) -> float:
    arr = to_numpy(values)
    return float(np.max(arr))


def median(values: Iterable[float]) -> float:
    arr = to_numpy(values)
    return float(np.median(arr))