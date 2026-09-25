"""Simple linear regression with inference on the slope."""

from __future__ import annotations

import math
from typing import Sequence

from . import distributions as d
from .descriptive import mean


def linear_regression(x: Sequence[float], y: Sequence[float], confidence: float = 0.95) -> dict[str, float]:
    """Least-squares fit y = a + b x, with standard errors, t test on the slope and R squared."""
    n = len(x)
    if n != len(y) or n < 3:
        raise ValueError("need at least 3 paired observations")
    mx, my = mean(x), mean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    slope = sxy / sxx
    intercept = my - slope * mx
    residuals = [b - (intercept + slope * a) for a, b in zip(x, y)]
    sse = sum(r * r for r in residuals)
    sst = sum((b - my) ** 2 for b in y)
    s2 = sse / (n - 2)
    se_slope = math.sqrt(s2 / sxx)
    se_intercept = math.sqrt(s2 * (1 / n + mx * mx / sxx))
    t = slope / se_slope
    tcrit = d.t_ppf(0.5 + confidence / 2, n - 2)
    return {
        "slope": slope, "intercept": intercept, "r2": 1 - sse / sst,
        "se_slope": se_slope, "se_intercept": se_intercept,
        "t_slope": t, "p_slope": 2 * (1 - d.t_cdf(abs(t), n - 2)),
        "slope_ci_low": slope - tcrit * se_slope, "slope_ci_high": slope + tcrit * se_slope,
        "residual_std": math.sqrt(s2),
    }
