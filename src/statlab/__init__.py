"""statlab: statistics implemented from first principles in pure Python."""

from . import distributions as dist
from .descriptive import describe, mean, median, mode, variance, std, quantile, iqr, skewness, kurtosis, covariance, correlation, frequency_table
from .inference import (
    TestResult, z_test, t_test_one_sample, t_test_two_sample, t_test_paired,
    chi2_independence, chi2_goodness_of_fit, proportion_ci, mean_ci, anova_one_way,
)
from .regression import linear_regression

__version__ = "0.1.0"
