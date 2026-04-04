import sys
from unittest.mock import patch

from faultmap.models import (
    AnalysisReport,
    ComparisonReport,
    CoverageGap,
    CoverageReport,
    FailureSlice,
    SliceComparison,
)
from faultmap.report import (
    _format_analysis_plain,
    _format_comparison_plain,
    _format_coverage_plain,
    format_analysis_report,
    format_comparison_report,
    format_coverage_report,
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
        root_cause="Model lacks legal domain training data",
        suggested_remediation="Add legal compliance context to system prompt",
    )


def _make_coverage_report(gaps=None, metadata=None):
    return CoverageReport(
        gaps=gaps or [],
        num_test_prompts=50,
        num_production_prompts=100,
        num_gaps=len(gaps) if gaps else 0,
        overall_coverage_score=1.0 if not gaps else 0.85,
        distance_threshold=0.5,
        embedding_model="test-embed",
        metadata=metadata or {},
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

    def test_with_slices_shows_insights(self):
        report = _make_analysis_report(slices=[_make_failure_slice()])
        text = format_analysis_report(report)
        assert "Model lacks legal domain training data" in text
        assert "Add legal compliance context to system prompt" in text

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

    def test_no_named_gaps_but_unclustered_prompts_are_reported(self):
        report = _make_coverage_report(
            metadata={"num_uncovered_total": 3, "num_unclustered_uncovered": 3}
        )
        text = format_coverage_report(report)
        assert "No reportable gap clusters found" in text

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

    def test_with_slices_shows_insights(self):
        report = _make_analysis_report(slices=[_make_failure_slice()])
        text = _format_analysis_plain(report)
        assert "Model lacks legal domain training data" in text
        assert "Add legal compliance context to system prompt" in text
        assert "Root Cause:" in text
        assert "Suggested Fix:" in text

    def test_with_slices_shows_examples(self):
        report = _make_analysis_report(slices=[_make_failure_slice()])
        text = _format_analysis_plain(report)
        assert "How do I comply with regulation X?" in text

    def test_no_insights_when_empty(self):
        """Root Cause / Suggested Fix lines are omitted when fields are empty strings."""
        s = FailureSlice(
            name="Simple", description="Simple desc",
            size=5, failure_rate=0.8, baseline_rate=0.1,
            effect_size=8.0, p_value=0.01, adjusted_p_value=0.02,
            test_used="chi2", sample_indices=[0],
            examples=[],
            representative_prompts=["A prompt"],
            cluster_id=0,
            # root_cause and suggested_remediation default to ""
        )
        report = _make_analysis_report(slices=[s])
        text = _format_analysis_plain(report)
        assert "Root Cause:" not in text
        assert "Suggested Fix:" not in text

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

    def test_shows_unclustered_metadata(self):
        report = _make_coverage_report(
            metadata={"num_uncovered_total": 3, "num_unclustered_uncovered": 3}
        )
        text = _format_coverage_plain(report)
        assert "Uncovered prompts:   3" in text
        assert "Unclustered:         3" in text
        assert "No reportable coverage gap clusters found" in text

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

    def test_comparison_falls_back_to_plain(self):
        report = _make_comparison_report()
        with patch.dict("sys.modules", self._null_rich()):
            text = format_comparison_report(report)
        assert "FAULTMAP MODEL COMPARISON REPORT" in text
        assert "=" * 10 in text


# ── Comparison fixtures ───────────────────────────────────


def _make_slice_comparison(winner: str = "a") -> SliceComparison:
    return SliceComparison(
        name="Legal Compliance",
        description="Questions about regulatory requirements",
        size=45,
        failure_rate_a=0.044,
        failure_rate_b=0.422,
        failure_rate_diff=-0.378,
        concordant_pass=24,
        concordant_fail=2,
        discordant_a_wins=18,
        discordant_b_wins=1,
        advantage_rate=0.947,
        p_value=0.0001,
        adjusted_p_value=0.0002,
        test_used="mcnemar_exact",
        winner=winner,
        sample_indices=list(range(45)),
        examples=[
            {
                "prompt": "How do I comply with GDPR?",
                "response_a": "resp_a",
                "response_b": "resp_b",
                "score_a": 0.9,
                "score_b": 0.2,
            }
        ],
        representative_prompts=["How do I comply with GDPR requirements?"],
        cluster_id=0,
    )


def _make_comparison_report(slices=None, global_winner: str = "a") -> ComparisonReport:
    return ComparisonReport(
        slices=slices if slices is not None else [],
        total_prompts=500,
        model_a_name="GPT-4o",
        model_b_name="GPT-4o-mini",
        failure_rate_a=0.12,
        failure_rate_b=0.24,
        global_p_value=0.0001,
        global_test_used="mcnemar_chi2",
        global_winner=global_winner,
        global_advantage_rate=0.78,
        significance_level=0.05,
        failure_threshold=0.5,
        scoring_mode="precomputed",
        num_clusters_tested=8,
        num_significant=len(slices) if slices else 0,
        clustering_method="hdbscan",
        embedding_model="text-embedding-3-small",
    )


# ── Comparison report rich tests ──────────────────────────


class TestFormatComparisonReportRich:
    """Tests via format_comparison_report() which tries rich first."""

    def test_comparison_no_significant_slices(self):
        report = _make_comparison_report()
        text = format_comparison_report(report)
        assert "No statistically significant per-slice differences found" in text

    def test_comparison_with_slices_contains_key_values(self):
        report = _make_comparison_report(slices=[_make_slice_comparison(winner="a")])
        text = format_comparison_report(report)
        assert "Legal Compliance" in text
        assert "0.95" in text or "0.94" in text  # advantage_rate 18/19 ≈ 0.947

    def test_comparison_global_shown(self):
        report = _make_comparison_report()
        text = format_comparison_report(report)
        assert "GPT-4o" in text
        assert "GPT-4o-mini" in text
        assert "0.0001" in text  # global_p_value

    def test_comparison_rich_returns_nonempty(self):
        report = _make_comparison_report()
        text = format_comparison_report(report)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_comparison_winner_shown_in_table(self):
        report = _make_comparison_report(slices=[_make_slice_comparison(winner="a")])
        text = format_comparison_report(report)
        assert "GPT-4o" in text  # winner column shows model name

    def test_comparison_failure_rates_shown(self):
        report = _make_comparison_report()
        text = format_comparison_report(report)
        # Rich may inject ANSI codes between "12.0" and "%"; check the value part
        assert "12.0" in text   # failure_rate_a
        assert "24.0" in text   # failure_rate_b


# ── Comparison report plain-text tests ────────────────────


class TestFormatComparisonPlain:
    """Direct calls to _format_comparison_plain bypass the rich try/except."""

    def test_no_slices_header(self):
        report = _make_comparison_report()
        text = _format_comparison_plain(report)
        assert "FAULTMAP MODEL COMPARISON REPORT" in text
        assert "No statistically significant per-slice differences found" in text

    def test_shows_metadata(self):
        report = _make_comparison_report()
        text = _format_comparison_plain(report)
        assert "GPT-4o" in text
        assert "GPT-4o-mini" in text
        assert "500" in text             # total_prompts
        assert "precomputed" in text     # scoring_mode
        assert "hdbscan" in text         # clustering_method
        assert "text-embedding-3-small" in text
        assert "GLOBAL COMPARISON" in text

    def test_global_winner_shown(self):
        report = _make_comparison_report(global_winner="a")
        text = _format_comparison_plain(report)
        assert "GPT-4o (Model A)" in text  # winner label

    def test_global_tie_shown(self):
        report = _make_comparison_report(global_winner="tie")
        text = _format_comparison_plain(report)
        assert "Winner:            tie" in text

    def test_advantage_rate_line(self):
        report = _make_comparison_report()
        text = _format_comparison_plain(report)
        assert "0.78" in text
        assert "78%" in text  # formatted as percentage
        assert "disagreements favor A" in text

    def test_with_slice_shows_slice_details(self):
        report = _make_comparison_report(slices=[_make_slice_comparison(winner="a")])
        text = _format_comparison_plain(report)
        assert "Legal Compliance" in text
        assert "Questions about regulatory requirements" in text
        assert "45 prompts" in text
        assert "mcnemar_exact" in text
        assert "A wins 18, B wins 1" in text

    def test_with_slice_winner_tag(self):
        report = _make_comparison_report(slices=[_make_slice_comparison(winner="a")])
        text = _format_comparison_plain(report)
        assert "** GPT-4o wins **" in text

    def test_with_slice_b_wins_tag(self):
        s = _make_slice_comparison(winner="b")
        report = _make_comparison_report(slices=[s])
        text = _format_comparison_plain(report)
        assert "** GPT-4o-mini wins **" in text

    def test_with_slice_shows_examples(self):
        report = _make_comparison_report(slices=[_make_slice_comparison(winner="a")])
        text = _format_comparison_plain(report)
        assert "How do I comply with GDPR requirements?" in text

    def test_comparison_plain_returns_nonempty(self):
        report = _make_comparison_report()
        text = _format_comparison_plain(report)
        assert isinstance(text, str)
        assert len(text) > 0
