import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from faultmap.analyzer import SliceAnalyzer
from faultmap.exceptions import ConfigurationError
from faultmap.models import ScoringResult


# ── Helpers ───────────────────────────────────────────────


def _make_analyzer(embedder):
    """Build a SliceAnalyzer with mocked internals (bypass __init__ embedder creation)."""
    analyzer = SliceAnalyzer.__new__(SliceAnalyzer)
    analyzer.significance_level = 0.05
    analyzer.min_slice_size = 5
    analyzer.failure_threshold = 0.5
    analyzer.clustering_method = "hdbscan"
    analyzer.embedding_model = "mock"
    analyzer._embedder = embedder
    analyzer._llm_client = AsyncMock()
    analyzer._llm_client.complete.return_value = (
        "Name: Test Slice\nDescription: Test description"
    )
    analyzer.n_samples = 8
    analyzer.temperature = 1.0
    analyzer.consistency_threshold = 0.8
    return analyzer


def _make_mock_embedder():
    """Hash-based deterministic MagicMock embedder."""
    embedder = MagicMock()
    embedder.dimension = 64

    def mock_embed(texts):
        embs = []
        for t in texts:
            seed = hash(t) % (2**31)
            rng = np.random.default_rng(seed)
            embs.append(rng.standard_normal(64))
        return np.array(embs, dtype=np.float32)

    embedder.embed = mock_embed
    return embedder


def _make_clustered_embeddings(n_clusters=3, n_per_cluster=30, dim=64, seed=42):
    """
    Generate well-separated embeddings suitable for HDBSCAN.
    Centers are scaled to unit-sphere * 5.0 with tight noise (0.15).
    """
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim))
    for i in range(n_clusters):
        centers[i] = centers[i] / np.linalg.norm(centers[i]) * 5.0
    parts = []
    for c in range(n_clusters):
        parts.append(centers[c] + rng.standard_normal((n_per_cluster, dim)) * 0.15)
    return np.vstack(parts).astype(np.float32)


def _make_coverage_embeddings(n_test=50, n_covered=30, n_gap=20, dim=64, seed=7):
    """Two-region embeddings: region A (test + covered prod), region B (gap prod)."""
    rng = np.random.default_rng(seed)
    center_a = rng.standard_normal(dim) * 5
    center_b = -center_a
    test_embs = (center_a + rng.standard_normal((n_test, dim)) * 0.2).astype(np.float32)
    prod_embs = np.vstack([
        center_a + rng.standard_normal((n_covered, dim)) * 0.2,
        center_b + rng.standard_normal((n_gap, dim)) * 0.2,
    ]).astype(np.float32)
    return test_embs, prod_embs


# ── __init__ validation ────────────────────────────────────


class TestSliceAnalyzerConfig:
    def test_invalid_clustering_method(self):
        with pytest.raises(ConfigurationError):
            SliceAnalyzer(clustering_method="invalid")

    def test_invalid_significance_level(self):
        with pytest.raises(ConfigurationError):
            SliceAnalyzer(significance_level=1.5)

    def test_invalid_n_samples(self):
        with pytest.raises(ConfigurationError):
            SliceAnalyzer(n_samples=1)

    def test_invalid_failure_threshold(self):
        with pytest.raises(ConfigurationError):
            SliceAnalyzer(failure_threshold=1.5)

    def test_init_stores_params(self):
        """Lines 55-70: __init__ attribute assignments + embedder/LLM construction."""
        with patch("faultmap.analyzer.get_embedder") as mock_get_embedder, \
             patch("faultmap.analyzer.AsyncLLMClient") as mock_llm_cls:
            analyzer = SliceAnalyzer(
                model="gpt-test",
                embedding_model="all-MiniLM-L6-v2",
                significance_level=0.1,
                min_slice_size=15,
                failure_threshold=0.4,
                n_samples=4,
                clustering_method="agglomerative",
                max_concurrent_requests=10,
                temperature=0.5,
                consistency_threshold=0.7,
            )

        assert analyzer.model == "gpt-test"
        assert analyzer.significance_level == 0.1
        assert analyzer.min_slice_size == 15
        assert analyzer.failure_threshold == 0.4
        assert analyzer.n_samples == 4
        assert analyzer.clustering_method == "agglomerative"
        assert analyzer.temperature == 0.5
        assert analyzer.consistency_threshold == 0.7
        mock_get_embedder.assert_called_once_with("all-MiniLM-L6-v2")
        mock_llm_cls.assert_called_once_with(
            model="gpt-test", max_concurrent_requests=10
        )


# ── Mode detection ────────────────────────────────────────


class TestAnalyzeMode1:
    def test_all_passing_no_slices(self):
        """All scores above threshold → 0 failures → early return."""
        analyzer = _make_analyzer(_make_mock_embedder())
        n = 100
        report = analyzer.analyze(
            [f"prompt-{i}" for i in range(n)],
            [f"response-{i}" for i in range(n)],
            scores=[0.9] * n,
        )
        assert report.scoring_mode == "precomputed"
        assert report.total_failures == 0
        assert len(report.slices) == 0

    def test_mode_detection_precomputed(self):
        analyzer = _make_analyzer(_make_mock_embedder())
        report = analyzer.analyze(
            [f"p-{i}" for i in range(20)],
            [f"r-{i}" for i in range(20)],
            scores=[0.9] * 20,
        )
        assert report.scoring_mode == "precomputed"

    def test_both_scores_and_references_warns(self):
        """Both provided → scores wins with UserWarning."""
        analyzer = _make_analyzer(_make_mock_embedder())
        with pytest.warns(UserWarning, match="Both scores and references"):
            report = analyzer.analyze(["a"], ["b"], scores=[0.9], references=["c"])
        assert report.scoring_mode == "precomputed"

    def test_report_carries_metadata(self):
        analyzer = _make_analyzer(_make_mock_embedder())
        n = 20
        report = analyzer.analyze(
            [f"prompt-{i}" for i in range(n)],
            [f"response-{i}" for i in range(n)],
            scores=[0.9] * n,
        )
        assert report.total_prompts == n
        assert report.clustering_method == "hdbscan"
        assert report.embedding_model == "mock"


class TestAnalyzeMode2:
    """Lines 138-140: elif references is not None → ReferenceScorer."""

    def test_mode_reference_detected(self):
        analyzer = _make_analyzer(_make_mock_embedder())
        mock_scorer = AsyncMock()
        mock_scorer.score.return_value = ScoringResult(
            scores=[0.9, 0.8], mode="reference"
        )
        with patch("faultmap.scoring.ReferenceScorer", return_value=mock_scorer):
            report = analyzer.analyze(["p1", "p2"], ["r1", "r2"], references=["ref1", "ref2"])
        assert report.scoring_mode == "reference"
        assert report.total_failures == 0

    def test_mode_reference_scorer_receives_embedder(self):
        """ReferenceScorer is instantiated with the analyzer's embedder."""
        embedder = _make_mock_embedder()
        analyzer = _make_analyzer(embedder)
        mock_scorer = AsyncMock()
        mock_scorer.score.return_value = ScoringResult(scores=[0.9], mode="reference")

        with patch("faultmap.scoring.ReferenceScorer", return_value=mock_scorer) as cls:
            analyzer.analyze(["p1"], ["r1"], references=["ref1"])
        # First positional arg should be the embedder
        cls.assert_called_once_with(embedder, ["ref1"])


class TestAnalyzeMode3:
    """Lines 141-147: else → EntropyScorer."""

    def test_mode_entropy_detected(self):
        analyzer = _make_analyzer(_make_mock_embedder())
        mock_scorer = AsyncMock()
        mock_scorer.score.return_value = ScoringResult(
            scores=[0.9, 0.8], mode="entropy"
        )
        with patch("faultmap.scoring.EntropyScorer", return_value=mock_scorer):
            report = analyzer.analyze(["p1", "p2"], ["r1", "r2"])
        assert report.scoring_mode == "entropy"
        assert report.total_failures == 0

    def test_mode_entropy_passes_correct_params(self):
        analyzer = _make_analyzer(_make_mock_embedder())
        mock_scorer = AsyncMock()
        mock_scorer.score.return_value = ScoringResult(scores=[0.9], mode="entropy")

        with patch("faultmap.scoring.EntropyScorer", return_value=mock_scorer) as cls:
            analyzer.analyze(["p1"], ["r1"])

        call_kwargs = cls.call_args[1]
        assert call_kwargs["n_samples"] == 8
        assert call_kwargs["temperature"] == 1.0
        assert call_kwargs["consistency_threshold"] == 0.8


# ── Full pipeline ─────────────────────────────────────────


class TestAnalyzeFullPipeline:
    """
    Tests that exercise lines 176-290 (embed → cluster → test → BH → assemble).
    Uses pre-computed well-separated embeddings so HDBSCAN is reliable.
    """

    def _make_clustered_analyzer(self, failure_clusters):
        """Analyzer whose embedder always returns the same well-separated embeddings."""
        n_clusters, n_per = 3, 30
        embeddings = _make_clustered_embeddings(n_clusters, n_per)

        topics = ["legal", "billing", "technical"]
        prompts = [
            f"[{topics[c]}] prompt {c}-{j}"
            for c in range(n_clusters)
            for j in range(n_per)
        ]
        responses = [f"response for {p}" for p in prompts]
        scores = [
            0.2 if c in failure_clusters else 0.8
            for c in range(n_clusters)
            for _ in range(n_per)
        ]

        embedder = MagicMock()
        embedder.dimension = 64
        embedder.embed = MagicMock(return_value=embeddings)

        analyzer = _make_analyzer(embedder)
        return analyzer, prompts, responses, scores

    def test_no_significant_slices_when_all_clusters_fail_equally(self):
        """
        When all clusters fail at the same rate as baseline, BH yields no significant.
        Hits the `if not significant: return AnalysisReport(slices=[], ...)` branch.
        """
        analyzer, prompts, responses, _ = self._make_clustered_analyzer([0, 1, 2])
        # All clusters fail → failure_rate == baseline → no cluster is more significant
        scores = [0.2] * len(prompts)
        report = analyzer.analyze(prompts, responses, scores=scores)
        assert report.total_failures == len(prompts)
        assert report.num_significant == 0
        assert len(report.slices) == 0

    def test_significant_slice_found_when_one_cluster_fails(self):
        """
        When cluster 0 fails at 100% vs ~33% baseline → highly significant.
        Hits lines 229-290: naming + FailureSlice assembly.
        """
        analyzer, prompts, responses, scores = self._make_clustered_analyzer([0])
        report = analyzer.analyze(prompts, responses, scores=scores)

        assert report.total_failures == 30
        assert report.baseline_failure_rate == pytest.approx(30 / 90)
        assert report.num_significant >= 1
        assert len(report.slices) >= 1

        worst = report.slices[0]
        assert worst.failure_rate > report.baseline_failure_rate
        assert worst.effect_size > 1.0
        assert worst.name == "Test Slice"
        assert worst.description == "Test description"
        assert len(worst.examples) > 0
        assert len(worst.representative_prompts) > 0

    def test_report_to_dict_is_serializable(self):
        analyzer, prompts, responses, scores = self._make_clustered_analyzer([0])
        report = analyzer.analyze(prompts, responses, scores=scores)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["num_significant"] >= 1
        assert isinstance(d["slices"], list)

    def test_scoring_metadata_carried_through(self):
        analyzer, prompts, responses, scores = self._make_clustered_analyzer([0])
        report = analyzer.analyze(prompts, responses, scores=scores)
        assert "scoring_metadata" in report.metadata

    def test_hdbscan_noise_labels_filtered(self):
        """
        Line 189: `unique_labels.remove(-1)` when HDBSCAN labels some points as noise.
        Add a single extreme outlier → HDBSCAN assigns it -1 (noise).
        """
        n_clusters, n_per = 3, 30
        base_embs = _make_clustered_embeddings(n_clusters, n_per)

        # Single outlier far from all clusters — HDBSCAN will mark it as noise (-1)
        outlier = np.full((1, 64), 100.0, dtype=np.float32)
        all_embs = np.vstack([base_embs, outlier])

        n_total = len(all_embs)
        n_base = len(base_embs)

        topics = ["legal", "billing", "technical"]
        prompts = (
            [f"[{topics[c]}] prompt {c}-{j}" for c in range(n_clusters) for j in range(n_per)]
            + ["outlier prompt"]
        )
        responses = [f"r-{i}" for i in range(n_total)]
        # Only cluster-0 fails; outlier passes
        scores = [0.2 if i < n_per else 0.8 for i in range(n_base)] + [0.8]

        embedder = MagicMock()
        embedder.dimension = 64
        embedder.embed = MagicMock(return_value=all_embs)

        analyzer = _make_analyzer(embedder)
        # Should not raise — noise label removal is handled transparently
        report = analyzer.analyze(prompts, responses, scores=scores)
        assert report.total_prompts == n_total


# ── Input validation ──────────────────────────────────────


class TestAnalyzeInputValidation:
    def test_empty_prompts_raises(self):
        analyzer = _make_analyzer(_make_mock_embedder())
        with pytest.raises(ConfigurationError):
            analyzer.analyze([], [], scores=[])

    def test_mismatched_lengths_raises(self):
        analyzer = _make_analyzer(_make_mock_embedder())
        with pytest.raises(ConfigurationError):
            analyzer.analyze(["a", "b"], ["r"], scores=[0.5, 0.5])

    def test_scores_out_of_range_raises(self):
        analyzer = _make_analyzer(_make_mock_embedder())
        with pytest.raises(ConfigurationError):
            analyzer.analyze(["a"], ["r"], scores=[1.5])


# ── audit_coverage ────────────────────────────────────────


class TestAuditCoverage:
    def test_empty_test_prompts_raises(self):
        analyzer = _make_analyzer(_make_mock_embedder())
        with pytest.raises(ConfigurationError):
            analyzer.audit_coverage([], ["prod-prompt"])

    def test_empty_production_prompts_raises(self):
        analyzer = _make_analyzer(_make_mock_embedder())
        with pytest.raises(ConfigurationError):
            analyzer.audit_coverage(["test-prompt"], [])

    def test_no_gaps_when_all_covered(self):
        """
        Lines 349-358: early return when unique_gaps is empty.
        High threshold forces all prod prompts to be 'covered'.
        """
        test_embs, prod_embs = _make_coverage_embeddings()
        n_test = len(test_embs)
        n_prod = len(prod_embs)

        embedder = MagicMock()
        embedder.dimension = 64
        embedder.embed.side_effect = [test_embs, prod_embs]

        analyzer = _make_analyzer(embedder)
        report = analyzer.audit_coverage(
            [f"t-{i}" for i in range(n_test)],
            [f"p-{i}" for i in range(n_prod)],
            distance_threshold=1000.0,  # everything is "covered"
        )

        assert report.num_gaps == 0
        assert report.overall_coverage_score == pytest.approx(1.0)
        assert report.num_test_prompts == n_test
        assert report.num_production_prompts == n_prod

    def test_gaps_found_and_named(self):
        """
        Lines 360-406: gap detection, representative extraction, LLM naming, assembly.
        Uses two-region embeddings with threshold 1.0 between covered (~0) and gap (~2).
        """
        test_embs, prod_embs = _make_coverage_embeddings(
            n_test=50, n_covered=30, n_gap=20
        )
        n_test, n_prod = len(test_embs), len(prod_embs)

        embedder = MagicMock()
        embedder.dimension = 64
        embedder.embed.side_effect = [test_embs, prod_embs]

        analyzer = _make_analyzer(embedder)
        report = analyzer.audit_coverage(
            [f"t-{i}" for i in range(n_test)],
            [f"p-{i}" for i in range(n_prod)],
            distance_threshold=1.0,   # separates covered (~0) from gap (~2)
            min_gap_size=5,
        )

        assert report.num_gaps >= 1
        assert report.overall_coverage_score < 1.0
        assert report.distance_threshold == pytest.approx(1.0)
        assert len(report.gaps) >= 1

        gap = report.gaps[0]
        assert gap.name == "Test Slice"  # from mock LLM
        assert gap.size >= 5
        assert gap.mean_distance > 0
        assert len(gap.representative_prompts) > 0

    def test_coverage_report_to_dict(self):
        test_embs, prod_embs = _make_coverage_embeddings()
        n_test, n_prod = len(test_embs), len(prod_embs)

        embedder = MagicMock()
        embedder.embed.side_effect = [test_embs, prod_embs]

        analyzer = _make_analyzer(embedder)
        report = analyzer.audit_coverage(
            [f"t-{i}" for i in range(n_test)],
            [f"p-{i}" for i in range(n_prod)],
            distance_threshold=1000.0,
        )
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "gaps" in d
        assert "overall_coverage_score" in d


# ── Report serialization ──────────────────────────────────


class TestReportSerializable:
    def test_analysis_report_to_dict(self):
        analyzer = _make_analyzer(_make_mock_embedder())
        report = analyzer.analyze(["a"] * 10, ["r"] * 10, scores=[0.9] * 10)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "slices" in d
        assert "total_prompts" in d

    def test_report_str(self):
        analyzer = _make_analyzer(_make_mock_embedder())
        report = analyzer.analyze(["a"] * 10, ["r"] * 10, scores=[0.9] * 10)
        text = str(report)
        assert isinstance(text, str)
        assert len(text) > 0
