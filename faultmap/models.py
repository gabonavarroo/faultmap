from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScoringResult:
    """Output of any scoring mode.

    Attributes:
        scores: Per-prompt quality scores in [0, 1]. Higher means better quality.
            A score below SliceAnalyzer.failure_threshold counts as a failure.
        mode: Scoring mode used: ``"precomputed"``, ``"reference"``, or ``"entropy"``.
        metadata: Mode-specific auxiliary data.
            - precomputed: ``{}``
            - reference: ``{"embedding_model": str}``
            - entropy: ``{"n_samples": int, "temperature": float,
              "semantic_entropy": list[float], "self_consistency": list[float],
              "normalized_entropy": list[float], "scores": list[float]}``
    """
    scores: list[float]           # Per-prompt score in [0, 1]. Higher = better.
    mode: str                     # "precomputed" | "reference" | "entropy"
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FailureSlice:
    """A single statistically significant failure cluster.

    Discovered by :meth:`SliceAnalyzer.analyze`.

    Attributes:
        name: LLM-generated human-readable name (e.g. ``"Legal compliance questions"``).
        description: LLM-generated one-sentence explanation of what this slice contains.
        size: Number of prompts belonging to this cluster.
        failure_rate: Fraction of prompts in this slice that failed (score < threshold).
        baseline_rate: Overall failure rate across all prompts, for comparison.
        effect_size: ``failure_rate / baseline_rate`` — how many times worse this slice
            is than average. ``2.0`` means twice the baseline failure rate.
        p_value: Raw p-value from the one-sided statistical test (chi-squared or Fisher).
        adjusted_p_value: Benjamini-Hochberg FDR-corrected p-value. The primary
            significance criterion: slices with ``adjusted_p_value < significance_level``
            are reported.
        test_used: Statistical test applied: ``"chi2"`` (chi-squared with Yates correction)
            or ``"fisher"`` (Fisher exact, used when expected cell count < 5).
        sample_indices: All indices in this cluster, referencing positions in the original
            ``prompts`` list passed to :meth:`~SliceAnalyzer.analyze`. Use these to recover
            the original data for a slice.
        examples: Up to 5 representative examples as dicts with keys
            ``"prompt"``, ``"response"``, and ``"score"``.
        representative_prompts: Up to 5 prompts closest to the cluster centroid,
            giving a quick intuition of what this slice is about.
        cluster_id: Internal HDBSCAN/agglomerative cluster label. Used for tracing.
    """
    name: str                          # LLM-generated name ("Legal compliance questions")
    description: str                   # LLM-generated 1-sentence explanation
    size: int                          # Number of prompts in this slice
    failure_rate: float                # Failure rate within this slice
    baseline_rate: float               # Overall failure rate for comparison
    effect_size: float                 # failure_rate / baseline_rate (risk ratio)
    p_value: float                     # Raw p-value from statistical test
    adjusted_p_value: float            # BH-corrected p-value
    test_used: str                     # "chi2" | "fisher"
    sample_indices: list[int]          # ALL indices in the cluster (into original prompts list)
    examples: list[dict]               # Top-5 example dicts with prompt/response/score
    representative_prompts: list[str]  # Top-5 prompts closest to cluster centroid
    cluster_id: int                    # Internal cluster label


@dataclass(frozen=True)
class AnalysisReport:
    """Complete output of :meth:`SliceAnalyzer.analyze`.

    Attributes:
        slices: Significant failure slices, sorted by ``adjusted_p_value`` ascending
            (most significant first).
        total_prompts: Total number of prompts analyzed.
        total_failures: Number of prompts with ``score < failure_threshold``.
        baseline_failure_rate: ``total_failures / total_prompts`` — the overall
            failure rate used as the comparison baseline for each cluster.
        significance_level: Alpha threshold used for Benjamini-Hochberg FDR correction.
        failure_threshold: Score cutoff used to binarize scores into pass/fail.
        scoring_mode: Which scoring mode was used: ``"precomputed"``, ``"reference"``,
            or ``"entropy"``.
        num_clusters_tested: Number of clusters that had at least ``min_slice_size``
            prompts and were submitted to statistical testing.
        num_significant: Number of slices that survived BH correction (``len(slices)``).
        clustering_method: Clustering algorithm used: ``"hdbscan"`` or ``"agglomerative"``.
        embedding_model: Name of the embedding model used to embed prompts.
        metadata: Optional auxiliary data. For entropy mode, includes
            ``"scoring_metadata"`` with per-prompt entropy and consistency scores.

    Example::

        report = analyzer.analyze(prompts, responses, scores=scores)
        print(report)                        # formatted output
        print(report.slices[0].name)         # "Legal compliance questions"
        print(report.slices[0].effect_size)  # 4.2
        d = report.to_dict()                 # JSON-serializable dict
    """
    slices: list[FailureSlice]    # Sorted by adjusted_p_value ascending
    total_prompts: int
    total_failures: int
    baseline_failure_rate: float
    significance_level: float
    failure_threshold: float
    scoring_mode: str             # "precomputed" | "reference" | "entropy"
    num_clusters_tested: int
    num_significant: int          # len(slices)
    clustering_method: str        # "hdbscan" | "agglomerative"
    embedding_model: str
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Return a one-paragraph plain-text summary of analysis results."""
        if not self.slices:
            return (
                f"No statistically significant failure slices found among "
                f"{self.total_prompts} prompts (baseline failure rate: "
                f"{self.baseline_failure_rate:.1%}, alpha={self.significance_level})."
            )
        worst = self.slices[0]
        return (
            f"Found {self.num_significant} significant failure slice(s) among "
            f"{self.total_prompts} prompts. Worst slice: \"{worst.name}\" "
            f"({worst.size} prompts, {worst.failure_rate:.1%} failure rate vs "
            f"{self.baseline_failure_rate:.1%} baseline, adj. p={worst.adjusted_p_value:.4f})."
        )

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary of all report fields.

        All nested dataclasses (``FailureSlice``) are recursively converted.
        Safe to pass to ``json.dumps()``.
        """
        from dataclasses import asdict
        return asdict(self)

    def __str__(self) -> str:
        from .report import format_analysis_report
        return format_analysis_report(self)


@dataclass(frozen=True)
class CoverageGap:
    """A single semantic gap discovered by :meth:`SliceAnalyzer.audit_coverage`.

    A gap is a cluster of production prompts that have no semantically similar
    prompt in the test suite (nearest-neighbor distance exceeds the threshold).

    Attributes:
        name: LLM-generated human-readable name for this gap cluster.
        description: LLM-generated one-sentence description of what topics are missing.
        size: Number of production prompts that fall into this gap.
        mean_distance: Average nearest-neighbor distance from gap prompts to the
            closest test prompt. Higher = farther from any test coverage.
        representative_prompts: Up to 5 production prompts closest to the gap
            cluster centroid. Useful for quickly understanding what's missing.
        prompt_indices: Indices of gap prompts into the original ``production_prompts``
            list passed to :meth:`~SliceAnalyzer.audit_coverage`.
        cluster_id: Internal cluster label for this gap.
    """
    name: str                     # LLM-generated name
    description: str              # LLM-generated description
    size: int                     # Number of production prompts in this gap
    mean_distance: float          # Average NN distance to nearest test prompt
    representative_prompts: list[str]  # Top-k production prompts
    prompt_indices: list[int]     # Indices into production_prompts list
    cluster_id: int


@dataclass(frozen=True)
class CoverageReport:
    """Complete output of :meth:`SliceAnalyzer.audit_coverage`.

    Attributes:
        gaps: Discovered coverage gaps, sorted by ``mean_distance`` descending
            (most severe gaps first).
        num_test_prompts: Number of test suite prompts analyzed.
        num_production_prompts: Number of production prompts analyzed.
        num_gaps: Number of gap clusters found (``len(gaps)``).
        overall_coverage_score: Fraction of production prompts that are covered
            (have a nearby test prompt). ``1.0`` = perfect coverage, ``0.8`` = 20%
            of production prompts are in gaps.
        distance_threshold: The nearest-neighbor distance threshold used to decide
            "covered" vs "uncovered". Auto-computed as ``mean + 1.5 * std`` if not
            specified.
        embedding_model: Name of the embedding model used.
        metadata: Optional auxiliary data. Coverage audits populate:
            ``"num_uncovered_total"``, ``"num_clustered_uncovered"``,
            ``"num_unclustered_uncovered"``, and
            ``"unclustered_prompt_indices"``.

    Example::

        coverage = analyzer.audit_coverage(test_prompts, prod_prompts)
        print(coverage)                             # formatted output
        print(coverage.overall_coverage_score)      # 0.82
        for gap in coverage.gaps:
            print(gap.name, gap.size)
    """
    gaps: list[CoverageGap]       # Sorted by mean_distance descending
    num_test_prompts: int
    num_production_prompts: int
    num_gaps: int                 # len(gaps)
    overall_coverage_score: float # 1 - (fraction of prod prompts in gaps)
    distance_threshold: float     # Threshold used for "uncovered"
    embedding_model: str
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        unclustered = int(self.metadata.get("num_unclustered_uncovered", 0))
        if not self.gaps:
            if unclustered:
                return (
                    f"No reportable coverage gap clusters found, but {unclustered} "
                    f"production prompt(s) are still uncovered and unclustered."
                )
            return (
                f"Test suite appears to cover production traffic well. "
                f"No significant gaps found among {self.num_production_prompts} "
                f"production prompts."
            )
        worst = self.gaps[0]
        return (
            f"Found {self.num_gaps} coverage gap(s). "
            f"Overall coverage score: {self.overall_coverage_score:.1%}. "
            f"Largest gap: \"{worst.name}\" ({worst.size} uncovered production prompts, "
            f"mean distance={worst.mean_distance:.3f})."
        )

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary of all report fields.

        All nested dataclasses (``CoverageGap``) are recursively converted.
        Safe to pass to ``json.dumps()``.
        """
        from dataclasses import asdict
        return asdict(self)

    def __str__(self) -> str:
        from .report import format_coverage_report
        return format_coverage_report(self)
