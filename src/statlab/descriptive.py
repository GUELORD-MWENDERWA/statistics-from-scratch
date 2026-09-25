"""Descriptive statistics on plain Python sequences."""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence


def _data(x: Sequence[float]) -> list[float]:
    values = [float(v) for v in x]
    if not values:
        raise ValueError("empty sample")
    return values


def mean(x: Sequence[float]) -> float:
    x = _data(x)
    return math.fsum(x) / len(x)


def median(x: Sequence[float]) -> float:
    return quantile(x, 0.5)


def mode(x: Sequence) -> list:
    counts = Counter(x)
    top = max(counts.values())
    return sorted(v for v, c in counts.items() if c == top)


def variance(x: Sequence[float], ddof: int = 1) -> float:
    """Sample variance (ddof=1, unbiased) or population variance (ddof=0)."""
    x = _data(x)
    if len(x) - ddof <= 0:
        raise ValueError("not enough data")
    m = mean(x)
    return math.fsum((v - m) ** 2 for v in x) / (len(x) - ddof)


def std(x: Sequence[float], ddof: int = 1) -> float:
    return math.sqrt(variance(x, ddof))


def quantile(x: Sequence[float], q: float) -> float:
    """Linear interpolation between order statistics (method 7, the default of R and NumPy)."""
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0, 1]")
    s = sorted(_data(x))
    h = (len(s) - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def iqr(x: Sequence[float]) -> float:
    return quantile(x, 0.75) - quantile(x, 0.25)


def skewness(x: Sequence[float]) -> float:
    """Fisher-Pearson coefficient (population moments)."""
    x = _data(x)
    m, s = mean(x), std(x, 0)
    return math.fsum(((v - m) / s) ** 3 for v in x) / len(x)


def kurtosis(x: Sequence[float]) -> float:
    """Excess kurtosis (0 for a normal distribution)."""
    x = _data(x)
    m, s = mean(x), std(x, 0)
    return math.fsum(((v - m) / s) ** 4 for v in x) / len(x) - 3


def covariance(x: Sequence[float], y: Sequence[float]) -> float:
    x, y = _data(x), _data(y)
    if len(x) != len(y):
        raise ValueError("samples must have the same length")
    mx, my = mean(x), mean(y)
    return math.fsum((a - mx) * (b - my) for a, b in zip(x, y)) / (len(x) - 1)


def correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation coefficient."""
    return covariance(x, y) / (std(x) * std(y))


def describe(x: Sequence[float]) -> dict[str, float]:
    x = _data(x)
    return {
        "n": len(x), "mean": mean(x), "std": std(x), "min": min(x),
        "q1": quantile(x, 0.25), "median": median(x), "q3": quantile(x, 0.75), "max": max(x),
        "skewness": skewness(x), "kurtosis": kurtosis(x),
    }


def frequency_table(x: Sequence[float], bins: int) -> list[tuple[float, float, int, float]]:
    """Equal-width class table: (lower, upper, count, relative frequency)."""
    x = _data(x)
    lo, hi = min(x), max(x)
    width = (hi - lo) / bins or 1.0
    counts = [0] * bins
    for v in x:
        counts[min(int((v - lo) / width), bins - 1)] += 1
    return [(lo + i * width, lo + (i + 1) * width, c, c / len(x)) for i, c in enumerate(counts)]
