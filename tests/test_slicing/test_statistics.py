from faultmap.slicing.statistics import (
    ClusterTestResult,
    _chi2_yates,
    _fisher_exact_one_sided,
    benjamini_hochberg,
    test_cluster_failure_rate,
)


class TestChi2Yates:
    def test_known_table(self):
        """
        Table:  Failed  Passed
        Cluster:  40      10     | 50
        Other:    40      110    | 150
                  80      120    | 200

        Expected failure rate in cluster: 40/50 = 80%
        Expected failure rate overall: 80/200 = 40%
        This should be highly significant.
        """
        p = _chi2_yates(a=40, b=10, c=40, d=110, n=200)
        assert p < 0.001  # Highly significant

    def test_no_difference(self):
        """Equal failure rates → p near 1.0."""
        # 20/50 in cluster vs 80/200 overall → both 40%
        p = _chi2_yates(a=20, b=30, c=60, d=90, n=200)
        assert p > 0.3  # Not significant


class TestFisherExact:
    def test_small_table_significant(self):
        """
        Table:  Failed  Passed
        Cluster:  7       1     | 8
        Other:    3       19    | 22
                  10      20    | 30
        """
        p = _fisher_exact_one_sided(a=7, b=1, c=3, d=19)
        assert p < 0.001

    def test_small_table_not_significant(self):
        """Proportional table → not significant."""
        # 3/8 in cluster vs 10/30 overall → ~same rate
        p = _fisher_exact_one_sided(a=3, b=5, c=7, d=15)
        assert p > 0.1


class TestClusterFailureRate:
    def test_higher_rate_uses_chi2(self):
        """Large enough expected counts → chi2."""
        result = test_cluster_failure_rate(
            cluster_failures=40, cluster_size=50,
            total_failures=80, total_size=200, cluster_id=0,
        )
        assert result.test_used == "chi2"
        assert result.p_value < 0.01
        assert result.failure_rate == 0.8

    def test_small_cluster_uses_fisher(self):
        """Small expected counts → Fisher."""
        result = test_cluster_failure_rate(
            cluster_failures=7, cluster_size=8,
            total_failures=10, total_size=30, cluster_id=0,
        )
        assert result.test_used == "fisher"
        assert result.p_value < 0.01

    def test_rate_not_higher_skips(self):
        """Cluster failure rate ≤ baseline → p = 1.0, test = 'none'."""
        result = test_cluster_failure_rate(
            cluster_failures=5, cluster_size=50,
            total_failures=80, total_size=200, cluster_id=0,
        )
        assert result.p_value == 1.0
        assert result.test_used == "none"


class TestBenjaminiHochberg:
    def test_known_correction(self):
        """Verify BH on known p-values."""
        results = [
            ClusterTestResult(i, 50, 10, 0.2, p, "chi2")
            for i, p in enumerate([0.001, 0.01, 0.03, 0.04, 0.06])
        ]
        corrected = benjamini_hochberg(results, alpha=0.05)

        adjusted = [r.adjusted_p_value for r in corrected]
        # First 4 should be ≤ 0.05, last should be > 0.05
        assert adjusted[0] <= 0.05   # 0.005
        assert adjusted[1] <= 0.05   # 0.025
        assert adjusted[2] <= 0.05   # 0.050
        assert adjusted[3] <= 0.05   # 0.050
        assert adjusted[4] > 0.05    # 0.060

    def test_empty(self):
        assert benjamini_hochberg([], alpha=0.05) == []

    def test_single_result(self):
        results = [ClusterTestResult(0, 50, 10, 0.2, 0.03, "chi2")]
        corrected = benjamini_hochberg(results, alpha=0.05)
        assert corrected[0].adjusted_p_value == 0.03  # m/rank = 1/1

    def test_monotonicity(self):
        """Adjusted p-values should be non-decreasing."""
        results = [
            ClusterTestResult(i, 50, 10, 0.2, p, "chi2")
            for i, p in enumerate([0.001, 0.005, 0.01, 0.05, 0.1, 0.5])
        ]
        corrected = benjamini_hochberg(results)
        adjusted = [r.adjusted_p_value for r in corrected]
        for i in range(len(adjusted) - 1):
            assert adjusted[i] <= adjusted[i + 1]
