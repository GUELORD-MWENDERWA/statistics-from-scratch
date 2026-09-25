"""Special functions needed by the distributions.

The regularised incomplete gamma and beta functions are evaluated with the
series and continued-fraction expansions described in Numerical Recipes
(Press et al.), chapter 6.
"""

from __future__ import annotations

import math

_EPS = 1e-15
_MAX_IT = 500


def gammainc(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a, x)."""
    if x < 0 or a <= 0:
        raise ValueError("need x >= 0 and a > 0")
    if x == 0:
        return 0.0
    if x < a + 1:
        # Series representation.
        term = total = 1.0 / a
        ap = a
        for _ in range(_MAX_IT):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * _EPS:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    return 1.0 - _gammainc_upper_cf(a, x)


def _gammainc_upper_cf(a: float, x: float) -> float:
    """Q(a, x) by Lentz's continued fraction."""
    tiny = 1e-300
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, _MAX_IT):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < _EPS:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1 / d
    h = d
    for m in range(1, _MAX_IT):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < _EPS:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if not 0 <= x <= 1:
        raise ValueError("x must be in [0, 1]")
    if x in (0.0, 1.0):
        return x
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1 - front * _betacf(b, a, 1 - x) / b


def invert(cdf, p: float, lo: float, hi: float, tol: float = 1e-12) -> float:
    """Invert a monotone CDF by bisection, expanding the bracket if needed."""
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    while cdf(lo) > p:
        lo = lo * 2 if lo < 0 else lo - 1
    while cdf(hi) < p:
        hi = hi * 2 if hi > 0 else hi + 1
    for _ in range(200):
        mid = (lo + hi) / 2
        if cdf(mid) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, abs(mid)):
            break
    return (lo + hi) / 2
