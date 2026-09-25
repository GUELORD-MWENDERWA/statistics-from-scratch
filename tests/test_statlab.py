import numpy as np
import pytest
from scipy import stats

import statlab as sl
from statlab import dist

RNG = np.random.default_rng(42)
X = list(RNG.normal(10, 2, 30))
Y = list(RNG.normal(11, 3, 25))


def test_descriptive_matches_numpy_and_scipy():
    assert sl.mean(X) == pytest.approx(np.mean(X))
    assert sl.std(X) == pytest.approx(np.std(X, ddof=1))
    for q in (0.1, 0.25, 0.5, 0.9):
        assert sl.quantile(X, q) == pytest.approx(np.quantile(X, q))
    assert sl.skewness(X) == pytest.approx(stats.skew(X))
    assert sl.kurtosis(X) == pytest.approx(stats.kurtosis(X))
    assert sl.correlation(X[:25], Y) == pytest.approx(np.corrcoef(X[:25], Y)[0, 1])
    assert sl.mode([1, 2, 2, 3, 3]) == [2, 3]


@pytest.mark.parametrize("x", [-3.2, -1, 0, 0.5, 2.7])
def test_continuous_cdfs(x):
    assert dist.norm_cdf(x) == pytest.approx(stats.norm.cdf(x), abs=1e-12)
    for df in (1, 3, 10, 50):
        assert dist.t_cdf(x, df) == pytest.approx(stats.t.cdf(x, df), abs=1e-10)
    for df in (1, 4, 15):
        assert dist.chi2_cdf(abs(x) * 3, df) == pytest.approx(stats.chi2.cdf(abs(x) * 3, df), abs=1e-10)
    assert dist.f_cdf(abs(x) + 0.1, 3, 12) == pytest.approx(stats.f.cdf(abs(x) + 0.1, 3, 12), abs=1e-10)


@pytest.mark.parametrize("p", [0.01, 0.05, 0.5, 0.975, 0.999])
def test_quantiles(p):
    assert dist.norm_ppf(p) == pytest.approx(stats.norm.ppf(p), abs=1e-8)
    assert dist.t_ppf(p, 7) == pytest.approx(stats.t.ppf(p, 7), abs=1e-7)
    assert dist.chi2_ppf(p, 5) == pytest.approx(stats.chi2.ppf(p, 5), abs=1e-7)


def test_discrete():
    assert dist.binom_pmf(3, 10, 0.3) == pytest.approx(stats.binom.pmf(3, 10, 0.3))
    assert dist.binom_cdf(4, 20, 0.25) == pytest.approx(stats.binom.cdf(4, 20, 0.25))
    assert dist.poisson_cdf(6, 3.5) == pytest.approx(stats.poisson.cdf(6, 3.5))


def test_t_tests_match_scipy():
    r = sl.t_test_one_sample(X, 9.5)
    ref = stats.ttest_1samp(X, 9.5)
    assert r.statistic == pytest.approx(ref.statistic) and r.p_value == pytest.approx(ref.pvalue)
    for equal in (True, False):
        r = sl.t_test_two_sample(X, Y, equal_var=equal)
        ref = stats.ttest_ind(X, Y, equal_var=equal)
        assert r.statistic == pytest.approx(ref.statistic) and r.p_value == pytest.approx(ref.pvalue)
    r = sl.t_test_paired(X[:25], Y, alternative="less")
    ref = stats.ttest_rel(X[:25], Y, alternative="less")
    assert r.p_value == pytest.approx(ref.pvalue)


def test_chi2_and_anova():
    table = [[20, 15, 25], [30, 35, 25]]
    r = sl.chi2_independence(table)
    ref = stats.chi2_contingency(table, correction=False)
    assert r.statistic == pytest.approx(ref.statistic) and r.p_value == pytest.approx(ref.pvalue)
    r = sl.chi2_goodness_of_fit([18, 22, 20, 25, 15])
    assert r.p_value == pytest.approx(stats.chisquare([18, 22, 20, 25, 15]).pvalue)
    a, b, c = X[:10], X[10:20], [v + 1.5 for v in X[20:]]
    r = sl.anova_one_way(a, b, c)
    ref = stats.f_oneway(a, b, c)
    assert r.statistic == pytest.approx(ref.statistic) and r.p_value == pytest.approx(ref.pvalue)


def test_confidence_intervals():
    lo, hi = sl.mean_ci(X)
    ref = stats.t.interval(0.95, len(X) - 1, loc=np.mean(X), scale=stats.sem(X))
    assert (lo, hi) == pytest.approx(ref)
    ref = stats.binomtest(42, 100).proportion_ci(0.95, method="wilson")
    assert sl.proportion_ci(42, 100) == pytest.approx((ref.low, ref.high))


def test_regression_matches_scipy():
    x = list(range(20))
    y = [2.5 * v + 3 + e for v, e in zip(x, RNG.normal(0, 2, 20))]
    r = sl.linear_regression(x, y)
    ref = stats.linregress(x, y)
    assert r["slope"] == pytest.approx(ref.slope)
    assert r["intercept"] == pytest.approx(ref.intercept)
    assert r["r2"] == pytest.approx(ref.rvalue**2)
    assert r["se_slope"] == pytest.approx(ref.stderr)
    assert r["p_slope"] == pytest.approx(ref.pvalue, abs=1e-12)
