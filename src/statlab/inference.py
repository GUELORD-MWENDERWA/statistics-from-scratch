"""Confidence intervals and hypothesis tests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from . import distributions as d
from .descriptive import mean, std, variance


@dataclass
class TestResult:
    statistic: float
    p_value: float
    df: float | None = None
    name: str = ""

    def reject(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


def _p_from_symmetric(stat: float, cdf, alternative: str) -> float:
    if alternative == "two-sided":
        return min(1.0, 2 * min(cdf(stat), 1 - cdf(stat)))
    if alternative == "greater":
        return 1 - cdf(stat)
    if alternative == "less":
        return cdf(stat)
    raise ValueError("alternative must be 'two-sided', 'greater' or 'less'")


def z_test(x: Sequence[float], mu0: float, sigma: float, alternative: str = "two-sided") -> TestResult:
    """One-sample z test with known population standard deviation."""
    z = (mean(x) - mu0) / (sigma / math.sqrt(len(x)))
    return TestResult(z, _p_from_symmetric(z, d.norm_cdf, alternative), name="z test")


def t_test_one_sample(x: Sequence[float], mu0: float, alternative: str = "two-sided") -> TestResult:
    n = len(x)
    t = (mean(x) - mu0) / (std(x) / math.sqrt(n))
    return TestResult(t, _p_from_symmetric(t, lambda v: d.t_cdf(v, n - 1), alternative), n - 1, "one-sample t test")


def t_test_two_sample(x: Sequence[float], y: Sequence[float], equal_var: bool = False, alternative: str = "two-sided") -> TestResult:
    """Student (pooled) or Welch two-sample t test."""
    nx, ny = len(x), len(y)
    vx, vy = variance(x), variance(y)
    if equal_var:
        df = nx + ny - 2
        sp2 = ((nx - 1) * vx + (ny - 1) * vy) / df
        se = math.sqrt(sp2 * (1 / nx + 1 / ny))
        name = "Student t test"
    else:
        se = math.sqrt(vx / nx + vy / ny)
        df = (vx / nx + vy / ny) ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
        name = "Welch t test"
    t = (mean(x) - mean(y)) / se
    return TestResult(t, _p_from_symmetric(t, lambda v: d.t_cdf(v, df), alternative), df, name)


def t_test_paired(x: Sequence[float], y: Sequence[float], alternative: str = "two-sided") -> TestResult:
    diffs = [a - b for a, b in zip(x, y)]
    r = t_test_one_sample(diffs, 0.0, alternative)
    r.name = "paired t test"
    return r


def mean_ci(x: Sequence[float], confidence: float = 0.95) -> tuple[float, float]:
    """t-based confidence interval for the mean."""
    n, m = len(x), mean(x)
    half = d.t_ppf(0.5 + confidence / 2, n - 1) * std(x) / math.sqrt(n)
    return m - half, m + half


def proportion_ci(successes: int, n: int, confidence: float = 0.95, method: str = "wilson") -> tuple[float, float]:
    """Confidence interval for a proportion (Wilson score by default, or Wald)."""
    p = successes / n
    z = d.norm_ppf(0.5 + confidence / 2)
    if method == "wald":
        half = z * math.sqrt(p * (1 - p) / n)
        return max(0.0, p - half), min(1.0, p + half)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def chi2_goodness_of_fit(observed: Sequence[float], expected: Sequence[float] | None = None) -> TestResult:
    if expected is None:
        expected = [sum(observed) / len(observed)] * len(observed)
    stat = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
    df = len(observed) - 1
    return TestResult(stat, 1 - d.chi2_cdf(stat, df), df, "chi-square goodness of fit")


def chi2_independence(table: Sequence[Sequence[float]]) -> TestResult:
    """Pearson chi-square test of independence on a contingency table (no continuity correction)."""
    rows = [sum(r) for r in table]
    cols = [sum(c) for c in zip(*table)]
    total = sum(rows)
    stat = 0.0
    for i, r in enumerate(table):
        for j, o in enumerate(r):
            e = rows[i] * cols[j] / total
            stat += (o - e) ** 2 / e
    df = (len(rows) - 1) * (len(cols) - 1)
    return TestResult(stat, 1 - d.chi2_cdf(stat, df), df, "chi-square independence")


def anova_one_way(*groups: Sequence[float]) -> TestResult:
    all_values = [v for g in groups for v in g]
    grand = mean(all_values)
    ss_between = sum(len(g) * (mean(g) - grand) ** 2 for g in groups)
    ss_within = sum(sum((v - mean(g)) ** 2 for v in g) for g in groups)
    df1, df2 = len(groups) - 1, len(all_values) - len(groups)
    f = (ss_between / df1) / (ss_within / df2)
    return TestResult(f, 1 - d.f_cdf(f, df1, df2), df1, "one-way ANOVA")
