from __future__ import annotations
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClusterTestResult:
    """Result of statistical test for one cluster."""
    cluster_id: int
    size: int
    failure_count: int
    failure_rate: float
    p_value: float
    test_used: str                # "chi2" | "fisher" | "none"
    adjusted_p_value: float = 1.0  # Set after BH correction


def test_cluster_failure_rate(
    cluster_failures: int,
    cluster_size: int,
    total_failures: int,
    total_size: int,
    cluster_id: int,
) -> ClusterTestResult:
    """
    Test if cluster's failure rate is significantly HIGHER than baseline.
    One-sided test.

    2x2 contingency table:
                        Failed    Passed
    In cluster:           a         b        | cluster_size
    Not in cluster:       c         d        | total_size - cluster_size
                        ----      ----
                      total_fail  total_pass  | total_size

    Decision: expected count < 5 in any cell → Fisher exact, else chi-squared.

    Edge cases:
    - failure_rate <= baseline → skip test, p = 1.0
    - cluster_size == total_size → p = 1.0 (cluster IS the whole dataset)
    - denominator == 0 → p = 1.0
    """
    a = cluster_failures
    b = cluster_size - cluster_failures
    c = total_failures - cluster_failures
    d = (total_size - cluster_size) - c

    failure_rate = a / cluster_size if cluster_size > 0 else 0.0
    baseline_rate = total_failures / total_size if total_size > 0 else 0.0

    # If cluster failure rate is not higher than baseline, skip
    if failure_rate <= baseline_rate:
        return ClusterTestResult(
            cluster_id=cluster_id,
            size=cluster_size,
            failure_count=a,
            failure_rate=failure_rate,
            p_value=1.0,
            test_used="none",
        )

    # Compute expected values to decide which test
    row_totals = [cluster_size, total_size - cluster_size]
    col_totals = [total_failures, total_size - total_failures]
    expected = np.array([
        [row_totals[0] * col_totals[0] / total_size,
         row_totals[0] * col_totals[1] / total_size],
        [row_totals[1] * col_totals[0] / total_size,
         row_totals[1] * col_totals[1] / total_size],
    ])

    if np.min(expected) < 5:
        p_value = _fisher_exact_one_sided(a, b, c, d)
        test_used = "fisher"
    else:
        p_value = _chi2_yates(a, b, c, d, total_size)
        test_used = "chi2"

    return ClusterTestResult(
        cluster_id=cluster_id,
        size=cluster_size,
        failure_count=a,
        failure_rate=failure_rate,
        p_value=p_value,
        test_used=test_used,
    )


# Prevent pytest from collecting this function as a test (name starts with "test_")
test_cluster_failure_rate.__test__ = False  # type: ignore[attr-defined]


def _chi2_yates(a: int, b: int, c: int, d: int, n: int) -> float:
    """
    Chi-squared with Yates' continuity correction for 2x2 table.

    Formula: chi2 = n * (|ad - bc| - n/2)² / ((a+b)(c+d)(a+c)(b+d))

    p-value for df=1: p = erfc(sqrt(chi2 / 2))
    This uses the exact relationship for chi2 with 1 degree of freedom.

    One-sided: divide two-sided p by 2 (we only care about higher rate).
    """
    from math import erfc, sqrt

    numerator = (abs(a * d - b * c) - n / 2.0) ** 2 * n
    denominator = (a + b) * (c + d) * (a + c) * (b + d)

    if denominator == 0:
        return 1.0

    chi2_stat = numerator / denominator
    p_two_sided = erfc(sqrt(chi2_stat / 2.0))
    return p_two_sided / 2.0


def _fisher_exact_one_sided(a: int, b: int, c: int, d: int) -> float:
    """
    Fisher exact test, one-sided (greater), without scipy.

    P(X >= a) where X ~ Hypergeometric(N=a+b+c+d, K=a+c, n=a+b).

    Uses log-gamma to avoid factorial overflow on large tables.
    """
    from math import lgamma, exp

    N = a + b + c + d
    K = a + c       # total failures
    n = a + b       # cluster size

    def log_choose(nn: int, kk: int) -> float:
        if kk < 0 or kk > nn:
            return float('-inf')
        return lgamma(nn + 1) - lgamma(kk + 1) - lgamma(nn - kk + 1)

    log_denom = log_choose(N, n)

    # Sum P(X = x) for x = a to min(n, K)
    max_x = min(n, K)
    p_value = 0.0
    for x in range(a, max_x + 1):
        log_p = log_choose(K, x) + log_choose(N - K, n - x) - log_denom
        p_value += exp(log_p)

    return min(p_value, 1.0)


def benjamini_hochberg(
    results: list[ClusterTestResult],
    alpha: float = 0.05,
) -> list[ClusterTestResult]:
    """
    Apply Benjamini-Hochberg FDR correction.

    Algorithm:
    1. Sort by p_value ascending
    2. adjusted[i] = p_value[i] * m / rank[i]  (rank is 1-indexed)
    3. Enforce monotonicity: backward pass, each = min(self, next)
    4. Clip to [0, 1]
    5. Assign to results

    Returns results sorted by adjusted_p_value ascending.

    Edge cases:
    - Empty list → return []
    - Single test → adjusted = raw p-value (m/rank = 1/1 = 1)
    - All p = 1.0 → all adjusted = 1.0
    """
    if not results:
        return []

    sorted_results = sorted(results, key=lambda r: r.p_value)
    m = len(sorted_results)

    # Raw adjusted p-values
    adjusted = [0.0] * m
    for i, result in enumerate(sorted_results):
        rank = i + 1
        adjusted[i] = result.p_value * m / rank

    # Enforce monotonicity (backward pass)
    for i in range(m - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    # Clip and assign
    for i, result in enumerate(sorted_results):
        result.adjusted_p_value = min(max(adjusted[i], 0.0), 1.0)

    return sorted(sorted_results, key=lambda r: r.adjusted_p_value)
