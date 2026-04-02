import sys
from unittest.mock import patch

import pytest

from faultmap.models import AnalysisReport, CoverageGap, CoverageReport, FailureSlice
from faultmap.report import (
    format_analysis_report,
    format_coverage_report,
    _format_analysis_plain,
    _format_coverage_plain,
)


# ── Fixtures ──────────────────────────────────────────────


def _make_analysis_report(slices=None):
    return AnalysisReport(
        slices=slices or [],
        total_prompts=100,
        total_failures=15,
        baseline_failure_rate=0.15,
        significance_level=0.05,
        failure_threshold=0.5,
        scoring_mode="precomputed",
        num_clusters_tested=5,
        num_significant=len(slices) if slices else 0,
        clustering_method="hdbscan",
        embedding_model="test-embed",
    )


def _make_failure_slice():
    return FailureSlice(
        name="Legal Questions",
        description="About legal compliance topics",
        size=20,
        failure_rate=0.8,
        baseline_rate=0.15,
        effect_size=5.3,
        p_value=0.001,
        adjusted_p_value=0.005,
        test_used="chi2",
        sample_indices=list(range(20)),
        examples=[{"prompt": "p", "response": "r", "score": 0.1}],
        representative_prompts=["How do I comply with regulation X?"],
        cluster_id=0,
    )


def _make_coverage_report(gaps=None):
    return CoverageReport(
        gaps=gaps or [],
        num_test_prompts=50,
        num_production_prompts=100,
        num_gaps=len(gaps) if gaps else 0,
        overall_coverage_score=1.0 if not gaps else 0.85,
        distance_threshold=0.5,
        embedding_model="test-embed",
    )


def _make_coverage_gap():
    return CoverageGap(
        name="Auth Gap",
        description="Two-factor authentication topics",
        size=15,
        mean_distance=1.8,
        representative_prompts=["How do I set up 2FA?", "Enable authenticator app"],
        prompt_indices=list(range(15)),
        cluster_id=0,
    )


# ── Rich formatting (default — rich is installed) ─────────


class TestFormatAnalysisReportRich:
    """Tests via format_analysis_report() which tries rich first."""

    def test_empty_slices(self):
        report = _make_analysis_report()
        text = format_analysis_report(report)
        assert "No statistically significant" in text
        assert "100" in text

    def test_with_slices_contains_key_values(self):
        report = _make_analysis_report(slices=[_make_failure_slice()])
        text = format_analysis_report(report)
        assert "Legal Questions" in text
        assert "80.0%" in text
        assert "5.3x" in text

    def test_returns_nonempty_string(self):
        report = _make_analysis_report()
        text = format_analysis_report(report)
        assert isinstance(text, str)
        assert len(text) > 0


class TestFormatCoverageReportRich:
    """Tests via format_coverage_report() which tries rich first."""

    def test_no_gaps(self):
        report = _make_coverage_report()
        text = format_coverage_report(report)
        assert "No significant coverage gaps" in text

    def test_with_gaps_contains_gap_name(self):
        """Hits the `if report.gaps:` table-rendering branch in _format_coverage_rich."""
        report = _make_coverage_report(gaps=[_make_coverage_gap()])
        text = format_coverage_report(report)
        assert "Auth Gap" in text

    def test_with_gaps_shows_counts(self):
        report = _make_coverage_report(gaps=[_make_coverage_gap()])
        text = format_coverage_report(report)
        assert "50" in text   # num_test_prompts
        assert "100" in text  # num_production_prompts


# ── Plain-text formatting (direct calls) ─────────────────


class TestFormatAnalysisPlain:
    """Direct calls to _format_analysis_plain bypass the rich try/except."""

    def test_empty_slices_header(self):
        report = _make_analysis_report()
        text = _format_analysis_plain(report)
        assert "FAULTMAP ANALYSIS REPORT" in text
        assert "No statistically significant" in text

    def test_shows_metadata(self):
        report = _make_analysis_report()
        text = _format_analysis_plain(report)
        assert "100" in text           # total_prompts
        assert "precomputed" in text   # scoring_mode
        assert "hdbscan" in text       # clustering_method
        assert "test-embed" in text    # embedding_model

    def test_with_slices_shows_slice_details(self):
        report = _make_analysis_report(slices=[_make_failure_slice()])
        text = _format_analysis_plain(report)
        assert "Legal Questions" in text
        assert "About legal compliance topics" in text
        assert "80.0%" in text
        assert "5.3x" in text
        assert "0.005000" in text   # adjusted_p_value
        assert "chi2" in text

    def test_with_slices_shows_examples(self):
        report = _make_analysis_report(slices=[_make_failure_slice()])
        text = _format_analysis_plain(report)
        assert "How do I comply with regulation X?" in text

    def test_prompt_truncated_at_120_chars(self):
        long_prompt = "A" * 150
        s = FailureSlice(
            name="Long", description="Desc",
            size=5, failure_rate=0.8, baseline_rate=0.1,
            effect_size=8.0, p_value=0.01, adjusted_p_value=0.02,
            test_used="chi2", sample_indices=[0],
            examples=[],
            representative_prompts=[long_prompt],
            cluster_id=0,
        )
        report = _make_analysis_report(slices=[s])
        text = _format_analysis_plain(report)
        assert "..." in text
        # The truncated prompt should appear
        assert long_prompt[:120] in text


class TestFormatCoveragePlain:
    """Direct calls to _format_coverage_plain."""

    def test_no_gaps_header(self):
        report = _make_coverage_report()
        text = _format_coverage_plain(report)
        assert "FAULTMAP COVERAGE REPORT" in text
        assert "No significant coverage gaps" in text

    def test_shows_metadata(self):
        report = _make_coverage_report()
        text = _format_coverage_plain(report)
        assert "50" in text           # num_test_prompts
        assert "100" in text          # num_production_prompts
        assert "100.0%" in text       # overall_coverage_score
        assert "test-embed" in text

    def test_with_gap_shows_gap_details(self):
        report = _make_coverage_report(gaps=[_make_coverage_gap()])
        text = _format_coverage_plain(report)
        assert "Auth Gap" in text
        assert "Two-factor authentication topics" in text
        assert "15" in text           # size
        assert "1.8000" in text       # mean_distance

    def test_with_gap_shows_examples(self):
        report = _make_coverage_report(gaps=[_make_coverage_gap()])
        text = _format_coverage_plain(report)
        assert "How do I set up 2FA?" in text


# ── ImportError fallback (rich absent) ───────────────────


class TestImportErrorFallback:
    """
    When rich is unavailable, format_* must fall back to plain text.
    We simulate missing rich by nulling its submodules in sys.modules.
    """

    def _null_rich(self):
        """Return a dict of all rich.* keys mapped to None."""
        return {k: None for k in list(sys.modules) if k == "rich" or k.startswith("rich.")}

    def test_analysis_falls_back_to_plain(self):
        report = _make_analysis_report()
        with patch.dict("sys.modules", self._null_rich()):
            text = format_analysis_report(report)
        # Plain text uses "=" separators, not rich markup
        assert "FAULTMAP ANALYSIS REPORT" in text
        assert "No statistically significant" in text
        assert "=" * 10 in text  # plain-text separator

    def test_analysis_with_slices_falls_back_to_plain(self):
        report = _make_analysis_report(slices=[_make_failure_slice()])
        with patch.dict("sys.modules", self._null_rich()):
            text = format_analysis_report(report)
        assert "Legal Questions" in text
        assert "80.0%" in text

    def test_coverage_falls_back_to_plain(self):
        report = _make_coverage_report()
        with patch.dict("sys.modules", self._null_rich()):
            text = format_coverage_report(report)
        assert "FAULTMAP COVERAGE REPORT" in text
        assert "No significant coverage gaps" in text

    def test_coverage_with_gaps_falls_back_to_plain(self):
        report = _make_coverage_report(gaps=[_make_coverage_gap()])
        with patch.dict("sys.modules", self._null_rich()):
            text = format_coverage_report(report)
        assert "FAULTMAP COVERAGE REPORT" in text
        assert "Auth Gap" in text
