from __future__ import annotations

import math

import pytest

from faultmap.comparison.statistics import (
    ComparisonTestResult,
    benjamini_hochberg_comparison,
    test_mcnemar,
)


class TestTestMcnemar:
    """Unit tests for test_mcnemar — covering both chi2 and exact paths."""

    def test_mcnemar_chi2_a_wins(self):
        """b=30, c=10 → b+c=40 ≥ 25 → chi2; A wins significantly."""
        # chi2 = max(20-1,0)² / 40 = 361/40 = 9.025
        # p = erfc(sqrt(9.025/2)) = erfc(2.124…) ≈ 0.000218
        result = test_mcnemar(30, 10, cluster_id=0, size=40)

        assert result.test_used == "mcnemar_chi2"
        assert result.p_value < 0.05
        assert result.advantage_rate == pytest.approx(30 / 40)
        assert result.winner == "a"
        assert result.b_count == 30
        assert result.c_count == 10
        assert result.cluster_id == 0
        assert result.size == 40

    def test_mcnemar_chi2_no_difference(self):
        """b=20, c=20 → b+c=40 ≥ 25 → chi2; advantage_rate=0.5 → tie; p=1.0."""
        # chi2 = max(0-1, 0)² / 40 = 0  →  p = erfc(0) = 1.0
        result = test_mcnemar(20, 20, cluster_id=1, size=40)

        assert result.test_used == "mcnemar_chi2"
        assert result.p_value == pytest.approx(1.0)
        assert result.advantage_rate == pytest.approx(0.5)
        assert result.winner == "tie"

    def test_mcnemar_exact_a_wins(self):
        """b=8, c=1 → b+c=9 < 25 → exact binomial; A wins significantly.

        P(X≥8 | Bin(9, 0.5)) = (C(9,8) + C(9,9)) / 2^9 = 10/512 ≈ 0.01953
        p = 2 × 0.01953 ≈ 0.03906 < 0.05
        """
        result = test_mcnemar(8, 1, cluster_id=2, size=9)

        assert result.test_used == "mcnemar_exact"
        assert result.p_value == pytest.approx(20 / 512, rel=1e-6)  # 2×10/512
        assert result.p_value < 0.05
        assert result.advantage_rate == pytest.approx(8 / 9)
        assert result.winner == "a"

    def test_mcnemar_exact_no_difference(self):
        """b=5, c=5 → b+c=10 < 25 → exact binomial; advantage_rate=0.5 → tie; p=1.0.

        P(X≥5 | Bin(10, 0.5)) = 638/1024 ≈ 0.623  →  p = min(1, 2×0.623) = 1.0
        """
        result = test_mcnemar(5, 5, cluster_id=3, size=10)

        assert result.test_used == "mcnemar_exact"
        assert result.p_value == pytest.approx(1.0)
        assert result.advantage_rate == pytest.approx(0.5)
        assert result.winner == "tie"

    def test_mcnemar_no_discordant_pairs(self):
        """b=0, c=0 → no discordant pairs → p=1.0, test='none', winner='tie'."""
        result = test_mcnemar(0, 0, cluster_id=4, size=30)

        assert result.p_value == pytest.approx(1.0)
        assert result.test_used == "none"
        assert result.winner == "tie"
        assert result.advantage_rate == pytest.approx(0.5)
        assert result.b_count == 0
        assert result.c_count == 0

    def test_mcnemar_all_discordant_one_direction(self):
        """b=15, c=0 → b+c=15 < 25 → exact; highly significant; A wins.

        P(X≥15 | Bin(15, 0.5)) = 1/2^15 = 1/32768
        p = 2/32768 ≈ 6.1e-5
        """
        result = test_mcnemar(15, 0, cluster_id=5, size=15)

        assert result.test_used == "mcnemar_exact"
        assert result.p_value == pytest.approx(2 / 2**15, rel=1e-6)
        assert result.p_value < 0.001
        assert result.winner == "a"
        assert result.advantage_rate == pytest.approx(1.0)

    def test_mcnemar_b_wins(self):
        """b=2, c=12 → b+c=14 < 25 → exact; B wins significantly.

        P(X≥12 | Bin(14, 0.5)) = (C(14,12)+C(14,13)+C(14,14)) / 2^14
                                = (91+14+1) / 16384 = 106/16384 ≈ 0.00647
        p = 2 × 0.00647 ≈ 0.01294 < 0.05
        """
        result = test_mcnemar(2, 12, cluster_id=6, size=14)

        assert result.test_used == "mcnemar_exact"
        assert result.p_value == pytest.approx(212 / 16384, rel=1e-6)  # 2×106/16384
        assert result.p_value < 0.05
        assert result.winner == "b"
        assert result.advantage_rate == pytest.approx(2 / 14)

    def test_mcnemar_threshold_exact_vs_chi2(self):
        """b+c=24 must use exact; b+c=25 must use chi2 (boundary check)."""
        # b=14, c=10 → b+c=24 → exact
        result_24 = test_mcnemar(14, 10, cluster_id=7, size=50)
        assert result_24.test_used == "mcnemar_exact"

        # b=15, c=10 → b+c=25 → chi2
        result_25 = test_mcnemar(15, 10, cluster_id=8, size=50)
        assert result_25.test_used == "mcnemar_chi2"

    def test_result_adjusted_p_value_default(self):
        """adjusted_p_value defaults to 1.0 before BH correction is applied."""
        result = test_mcnemar(10, 2, cluster_id=0, size=12)
        assert result.adjusted_p_value == pytest.approx(1.0)

    def test_is_not_pytest_test(self):
        """test_mcnemar must carry __test__ = False to suppress pytest collection."""
        assert test_mcnemar.__test__ is False  # type: ignore[attr-defined]


class TestBenjaminiHochbergComparison:
    """Unit tests for benjamini_hochberg_comparison."""

    def _make_result(self, cluster_id: int, p_value: float) -> ComparisonTestResult:
        return ComparisonTestResult(
            cluster_id=cluster_id,
            size=30,
            b_count=8,
            c_count=2,
            advantage_rate=0.8,
            p_value=p_value,
            test_used="mcnemar_exact",
            winner="a",
        )

    def test_bh_correction_comparison(self):
        """Known p-values → verify adjusted values exactly.

        p-values (unsorted input): [0.20, 0.08, 0.01, 0.04]
        Sorted rank:               [0.01, 0.04, 0.08, 0.20]  (m=4)

        Raw BH adjusted:
          rank 1: 0.01 × 4/1 = 0.04
          rank 2: 0.04 × 4/2 = 0.08
          rank 3: 0.08 × 4/3 ≈ 0.10667
          rank 4: 0.20 × 4/4 = 0.20

        After backward monotonicity pass: [0.04, 0.08, 0.10667, 0.20]
        """
        results = [
            self._make_result(i, p)
            for i, p in enumerate([0.20, 0.08, 0.01, 0.04])  # intentionally unsorted
        ]

        corrected = benjamini_hochberg_comparison(results, alpha=0.05)

        assert len(corrected) == 4
        adj = [r.adjusted_p_value for r in corrected]

        assert adj[0] == pytest.approx(0.04, abs=1e-9)
        assert adj[1] == pytest.approx(0.08, abs=1e-9)
        assert adj[2] == pytest.approx(4 * 0.08 / 3, rel=1e-9)   # ≈ 0.10667
        assert adj[3] == pytest.approx(0.20, abs=1e-9)

    def test_bh_comparison_empty(self):
        """Empty input returns empty list."""
        assert benjamini_hochberg_comparison([]) == []

    def test_bh_comparison_single(self):
        """Single result: adjusted equals raw p-value (m/rank = 1/1 = 1)."""
        result = self._make_result(0, 0.03)
        corrected = benjamini_hochberg_comparison([result])

        assert len(corrected) == 1
        assert corrected[0].adjusted_p_value == pytest.approx(0.03)

    def test_bh_comparison_monotonicity(self):
        """Adjusted p-values returned must be non-decreasing."""
        p_values = [0.001, 0.003, 0.01, 0.05, 0.10, 0.30, 0.80]
        results = [self._make_result(i, p) for i, p in enumerate(p_values)]

        corrected = benjamini_hochberg_comparison(results)
        adj = [r.adjusted_p_value for r in corrected]

        for i in range(len(adj) - 1):
            assert adj[i] <= adj[i + 1], (
                f"Monotonicity violated at index {i}: "
                f"adj[{i}]={adj[i]:.6f} > adj[{i+1}]={adj[i+1]:.6f}"
            )

    def test_bh_comparison_all_ones(self):
        """All p-values 1.0 → all adjusted values remain 1.0."""
        results = [self._make_result(i, 1.0) for i in range(5)]
        corrected = benjamini_hochberg_comparison(results)

        for r in corrected:
            assert r.adjusted_p_value == pytest.approx(1.0)

    def test_bh_comparison_sorted_output(self):
        """Output is sorted by adjusted_p_value ascending."""
        results = [
            self._make_result(0, 0.50),
            self._make_result(1, 0.01),
            self._make_result(2, 0.10),
        ]
        corrected = benjamini_hochberg_comparison(results)
        adj = [r.adjusted_p_value for r in corrected]

        assert adj == sorted(adj)

    def test_bh_comparison_clips_to_one(self):
        """Adjusted p-values must not exceed 1.0."""
        # With a single very large p-value, BH should clip at 1.0
        results = [self._make_result(i, p) for i, p in enumerate([0.8, 0.9, 1.0])]
        corrected = benjamini_hochberg_comparison(results)

        for r in corrected:
            assert r.adjusted_p_value <= 1.0
