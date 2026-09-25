"""Probability distributions: pdf/pmf, cdf and ppf (quantile function)."""

from __future__ import annotations

import math

from .special import betainc, gammainc, invert

# Normal -----------------------------------------------------------------


def norm_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2 * math.pi))


def norm_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    return 0.5 * math.erfc(-(x - mu) / (sigma * math.sqrt(2)))


def norm_ppf(p: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    return mu + sigma * invert(norm_cdf, p, -10, 10)


# Student t --------------------------------------------------------------


def t_pdf(x: float, df: float) -> float:
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    return c * (1 + x * x / df) ** (-(df + 1) / 2)


def t_cdf(x: float, df: float) -> float:
    tail = 0.5 * betainc(df / 2, 0.5, df / (df + x * x))
    return 1 - tail if x > 0 else tail


def t_ppf(p: float, df: float) -> float:
    return invert(lambda x: t_cdf(x, df), p, -50, 50)


# Chi-square -------------------------------------------------------------


def chi2_cdf(x: float, df: float) -> float:
    return 0.0 if x <= 0 else gammainc(df / 2, x / 2)


def chi2_ppf(p: float, df: float) -> float:
    return invert(lambda x: chi2_cdf(x, df), p, 0.0, max(10.0, 5 * df))


# Fisher F ---------------------------------------------------------------


def f_cdf(x: float, d1: float, d2: float) -> float:
    return 0.0 if x <= 0 else betainc(d1 / 2, d2 / 2, d1 * x / (d1 * x + d2))


def f_ppf(p: float, d1: float, d2: float) -> float:
    return invert(lambda x: f_cdf(x, d1, d2), p, 0.0, 50.0)


# Discrete ---------------------------------------------------------------


def binom_pmf(k: int, n: int, p: float) -> float:
    if not 0 <= k <= n:
        return 0.0
    return math.comb(n, k) * p**k * (1 - p) ** (n - k)


def binom_cdf(k: int, n: int, p: float) -> float:
    return sum(binom_pmf(i, n, p) for i in range(0, min(k, n) + 1)) if k >= 0 else 0.0


def poisson_pmf(k: int, lam: float) -> float:
    return 0.0 if k < 0 else math.exp(k * math.log(lam) - lam - math.lgamma(k + 1)) if lam > 0 else float(k == 0)


def poisson_cdf(k: int, lam: float) -> float:
    return sum(poisson_pmf(i, lam) for i in range(0, k + 1)) if k >= 0 else 0.0


def expon_cdf(x: float, lam: float) -> float:
    return 0.0 if x < 0 else 1 - math.exp(-lam * x)
