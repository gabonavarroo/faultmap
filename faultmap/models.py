from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScoringResult:
    """Output of any scoring mode."""
    scores: list[float]           # Per-prompt score in [0, 1]. Higher = better.
    mode: str                     # "precomputed" | "reference" | "entropy"
    metadata: dict = field(default_factory=dict)
    # metadata varies by mode:
    #   precomputed: {}
    #   reference:   {"embedding_model": str}
    #   entropy:     {"n_samples": int, "temperature": float,
    #                 "semantic_entropy": list[float],
    #                 "self_consistency": list[float],
    #                 "normalized_entropy": list[float]}


@dataclass(frozen=True)
class FailureSlice:
    """A single discovered failure cluster."""
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
    examples: list[dict]               # Top-5 dicts: {"prompt": str, "response": str, "score": float}
    representative_prompts: list[str]  # Top-5 prompts closest to cluster centroid
    cluster_id: int                    # Internal cluster label


@dataclass(frozen=True)
class AnalysisReport:
    """Complete output of SliceAnalyzer.analyze()."""
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
        """Return a one-paragraph plain-text summary."""
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
        """Return a JSON-serializable dictionary."""
        from dataclasses import asdict
        return asdict(self)

    def __str__(self) -> str:
        from .report import format_analysis_report
        return format_analysis_report(self)


@dataclass(frozen=True)
class CoverageGap:
    """A single discovered gap in test coverage."""
    name: str                     # LLM-generated name
    description: str              # LLM-generated description
    size: int                     # Number of production prompts in this gap
    mean_distance: float          # Average NN distance to nearest test prompt
    representative_prompts: list[str]  # Top-k production prompts
    prompt_indices: list[int]     # Indices into production_prompts list
    cluster_id: int


@dataclass(frozen=True)
class CoverageReport:
    """Complete output of SliceAnalyzer.audit_coverage()."""
    gaps: list[CoverageGap]       # Sorted by mean_distance descending
    num_test_prompts: int
    num_production_prompts: int
    num_gaps: int                 # len(gaps)
    overall_coverage_score: float # 1 - (fraction of prod prompts in gaps)
    distance_threshold: float     # Threshold used for "uncovered"
    embedding_model: str
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        if not self.gaps:
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
        from dataclasses import asdict
        return asdict(self)

    def __str__(self) -> str:
        from .report import format_coverage_report
        return format_coverage_report(self)
