from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ComparisonTestResult:
    """Result of McNemar's test for one cluster (or the global population).

    Mutable so that :func:`benjamini_hochberg_comparison` can set
    ``adjusted_p_value`` in-place after sorting, following the same pattern
    as :class:`~faultmap.slicing.statistics.ClusterTestResult`.

    Attributes:
        cluster_id: Cluster label (``-1`` for the global test).
        size: Total prompts in the cluster.
        b_count: Discordant A-wins — Model A passed, Model B failed.
        c_count: Discordant B-wins — Model A failed, Model B passed.
        advantage_rate: ``b / (b+c)`` — proportion of disagreements where A
            wins. ``0.5`` when there are no discordant pairs.
        p_value: Raw two-sided McNemar p-value.
        test_used: ``"mcnemar_chi2"`` | ``"mcnemar_exact"`` | ``"none"``.
        winner: Directional winner based on ``advantage_rate``: ``"a"`` when
            ``> 0.5``, ``"b"`` when ``< 0.5``, ``"tie"`` when exactly ``0.5``
            or no discordant pairs. Note: this does **not** account for the
            significance threshold — the final :class:`~faultmap.models.SliceComparison`
            winner is set by the analyzer after BH correction.
        adjusted_p_value: Benjamini-Hochberg corrected p-value. Defaults to
            ``1.0`` and is updated by :func:`benjamini_hochberg_comparison`.
    """

    cluster_id: int
    size: int
    b_count: int                    # discordant A-wins (A pass, B fail)
    c_count: int                    # discordant B-wins (A fail, B pass)
    advantage_rate: float           # b/(b+c); 0.5 when no discordant pairs
    p_value: float
    test_used: str                  # "mcnemar_chi2" | "mcnemar_exact" | "none"
    winner: str                     # "a" | "b" | "tie" (directional, pre-BH)
    adjusted_p_value: float = field(default=1.0)  # set by BH correction


def test_mcnemar(
    b_count: int,
    c_count: int,
    cluster_id: int,
    size: int,
) -> ComparisonTestResult:
    """Run McNemar's test on a set of paired binary outcomes.

    McNemar's test is the correct test for paired binary data. Only the
    **discordant pairs** (where the models disagree) carry information:

    - ``b_count`` (cell b): Model A passes, Model B fails  → "A wins"
    - ``c_count`` (cell c): Model A fails, Model B passes  → "B wins"

    **Test selection**:

    - ``b + c >= 25``: chi-squared approximation with Edwards' continuity
      correction — ``chi2 = max(|b−c|−1, 0)² / (b+c)``, then
      ``p = erfc(sqrt(chi2/2))``.  Uses the same ``erfc(sqrt(·/2))`` identity
      already in :mod:`faultmap.slicing.statistics`.
    - ``b + c < 25``: exact two-sided binomial test — under H₀,
      ``b ~ Binomial(b+c, 0.5)``, so ``p = min(1, 2·P(X ≥ max(b,c)))``.
    - ``b + c == 0``: no discordant pairs; ``p = 1.0``, ``test_used = "none"``.

    Args:
        b_count: Number of prompts where A passed and B failed.
        c_count: Number of prompts where A failed and B passed.
        cluster_id: Cluster identifier (pass ``-1`` for the global test).
        size: Total prompts in the cluster (including concordant pairs).

    Returns:
        :class:`ComparisonTestResult` with raw p-value, test variant, and
        directional ``winner`` based on ``advantage_rate`` alone (before any
        significance threshold is applied).
    """
    n_discordant = b_count + c_count

    if n_discordant == 0:
        return ComparisonTestResult(
            cluster_id=cluster_id,
            size=size,
            b_count=0,
            c_count=0,
            advantage_rate=0.5,
            p_value=1.0,
            test_used="none",
            winner="tie",
        )

    advantage_rate = b_count / n_discordant

    if n_discordant >= 25:
        p_value = _mcnemar_chi2(b_count, c_count)
        test_used = "mcnemar_chi2"
    else:
        p_value = _exact_binomial_two_sided(b_count, c_count)
        test_used = "mcnemar_exact"

    if advantage_rate > 0.5:
        winner = "a"
    elif advantage_rate < 0.5:
        winner = "b"
    else:
        winner = "tie"

    return ComparisonTestResult(
        cluster_id=cluster_id,
        size=size,
        b_count=b_count,
        c_count=c_count,
        advantage_rate=advantage_rate,
        p_value=p_value,
        test_used=test_used,
        winner=winner,
    )


# Prevent pytest from collecting this function as a test (name starts with "test_")
test_mcnemar.__test__ = False  # type: ignore[attr-defined]


def _mcnemar_chi2(b: int, c: int) -> float:
    """Two-sided McNemar chi-squared p-value with Edwards' continuity correction.

    Formula::

        corrected = max(|b - c| - 1, 0)
        chi2      = corrected² / (b + c)
        p         = erfc(sqrt(chi2 / 2))   # exact for df=1

    The ``erfc(sqrt(chi2/2))`` identity gives the **exact** two-sided p-value
    for a chi-squared distribution with 1 degree of freedom, without importing
    any external statistical library.

    Args:
        b: Discordant A-wins count.
        c: Discordant B-wins count.

    Returns:
        Two-sided p-value in ``[0, 1]``. Returns ``1.0`` when ``b + c == 0``.
    """
    from math import erfc, sqrt

    n = b + c
    if n == 0:
        return 1.0

    # Edwards' continuity correction: subtract 1 from |b - c|, floor at 0
    corrected = max(abs(b - c) - 1, 0)
    chi2 = (corrected ** 2) / n
    return erfc(sqrt(chi2 / 2.0))


def _exact_binomial_two_sided(b: int, c: int) -> float:
    """Two-sided exact binomial p-value for McNemar's test (small-sample case).

    Under H₀, ``b ~ Binomial(n=b+c, p=0.5)`` — each discordant pair is equally
    likely to favor either model. The two-sided p-value uses the more extreme
    tail::

        p = min(1, 2 · P(X ≥ max(b, c)))   where X ~ Bin(b+c, 0.5)

    ``P(X ≥ k)`` is computed via log-gamma to avoid factorial overflow, using
    the same ``lgamma``-based technique as
    :func:`~faultmap.slicing.statistics._fisher_exact_one_sided`.

    Args:
        b: Discordant A-wins count.
        c: Discordant B-wins count.

    Returns:
        Two-sided p-value in ``[0, 1]``. Returns ``1.0`` when ``b + c == 0``.
    """
    from math import exp, lgamma, log

    n = b + c
    if n == 0:
        return 1.0

    # Use the larger count to identify the more extreme tail
    k = max(b, c)

    # log P(X = x) = log C(n, x) + n · log(0.5)
    log_half_n = n * log(0.5)  # = -n · log(2)

    def log_choose(nn: int, kk: int) -> float:
        if kk < 0 or kk > nn:
            return float("-inf")
        return lgamma(nn + 1) - lgamma(kk + 1) - lgamma(nn - kk + 1)

    # P(X >= k) = sum P(X = x) for x in [k, n]
    p_tail = 0.0
    for x in range(k, n + 1):
        p_tail += exp(log_choose(n, x) + log_half_n)

    # Two-sided: multiply by 2 (Binomial(n, 0.5) is symmetric)
    return min(1.0, 2.0 * p_tail)


def benjamini_hochberg_comparison(
    results: list[ComparisonTestResult],
    alpha: float = 0.05,
) -> list[ComparisonTestResult]:
    """Apply Benjamini-Hochberg FDR correction to a list of McNemar test results.

    Parallel to :func:`~faultmap.slicing.statistics.benjamini_hochberg` but
    operates on :class:`ComparisonTestResult` objects instead of
    :class:`~faultmap.slicing.statistics.ClusterTestResult`, keeping the two
    type hierarchies separate and the existing tested interface unchanged.

    Algorithm:

    1. Sort by ``p_value`` ascending.
    2. ``adjusted[i] = p_value[i] · m / rank[i]``  (rank is 1-indexed).
    3. Enforce monotonicity via backward pass: ``adjusted[i] = min(adjusted[i], adjusted[i+1])``.
    4. Clip each adjusted value to ``[0, 1]`` and assign to ``result.adjusted_p_value``.
    5. Return results sorted by ``adjusted_p_value`` ascending.

    Args:
        results: List of :class:`ComparisonTestResult` objects to correct.
            Modified in-place (``adjusted_p_value`` is set on each object).
        alpha: Significance threshold (unused in the correction itself but
            kept for API symmetry with the slicing counterpart).

    Returns:
        The same list, sorted by ``adjusted_p_value`` ascending.
        Returns ``[]`` for empty input.

    Edge cases:

    - Empty list → returns ``[]``.
    - Single test → adjusted equals raw p-value (``m/rank = 1``).
    - All p-values ``1.0`` → all adjusted remain ``1.0``.
    """
    if not results:
        return []

    sorted_results = sorted(results, key=lambda r: r.p_value)
    m = len(sorted_results)

    # Step 2: raw BH adjustment
    adjusted = [0.0] * m
    for i, result in enumerate(sorted_results):
        rank = i + 1
        adjusted[i] = result.p_value * m / rank

    # Step 3: enforce monotonicity (backward pass)
    for i in range(m - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    # Step 4: clip to [0, 1] and assign
    for i, result in enumerate(sorted_results):
        result.adjusted_p_value = min(max(adjusted[i], 0.0), 1.0)

    # Step 5: return sorted by adjusted p-value
    return sorted(sorted_results, key=lambda r: r.adjusted_p_value)
