# Phase 5 — Integration (Day 5)

**Goal**: Wire everything together — coverage detection, report formatting, and the main `SliceAnalyzer` orchestrator.

**Files to create**:
- `faultmap/coverage/__init__.py`
- `faultmap/coverage/detector.py`
- `faultmap/report.py`
- `faultmap/analyzer.py`
- `tests/test_coverage/test_detector.py`
- `tests/test_report.py`
- `tests/test_analyzer.py`

**Milestone**: Full `analyzer.analyze()` and `analyzer.audit_coverage()` work end-to-end.

---

## 1. `coverage/detector.py`

Nearest-neighbor coverage gap detection.

```python
from __future__ import annotations
import logging

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..exceptions import ClusteringError, ConfigurationError
from ..slicing.clustering import cluster_embeddings, get_representative_prompts

logger = logging.getLogger(__name__)


def detect_coverage_gaps(
    test_embeddings: np.ndarray,
    prod_embeddings: np.ndarray,
    prod_prompts: list[str],
    distance_threshold: float | None = None,
    min_gap_size: int = 5,
    clustering_method: str = "hdbscan",
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Find production prompts far from any test prompt.

    Algorithm:
    1. L2-normalize both test and prod embeddings
    2. Fit NearestNeighbors(k=1) on test embeddings
    3. Query with prod embeddings → distances array
    4. Auto-threshold if None: mean + 1.5 * std of distances
    5. Uncovered = distance > threshold
    6. If enough uncovered: cluster them into coherent gap groups
    7. Map cluster labels back to full prod array

    Args:
        test_embeddings: (n_test, d)
        prod_embeddings: (n_prod, d)
        prod_prompts: original production prompt strings
        distance_threshold: custom threshold, or None for auto
        min_gap_size: minimum uncovered prompts to form a gap
        clustering_method: "hdbscan" or "agglomerative"

    Returns:
        gap_labels: (n_prod,) -1=covered, >=0=gap cluster id
        nn_distances: (n_prod,) nearest-neighbor distances
        distance_threshold: the threshold that was used

    Edge cases:
    - No test prompts → ConfigurationError
    - No prod prompts → return empty arrays
    - All covered → valid result (all -1)
    - Too few uncovered to cluster → treat all uncovered as one gap
    """
    if test_embeddings.shape[0] == 0:
        raise ConfigurationError("test_embeddings must be non-empty")
    if prod_embeddings.shape[0] == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=float),
            distance_threshold or 0.0,
        )

    # L2-normalize
    test_norms = np.linalg.norm(test_embeddings, axis=1, keepdims=True) + 1e-10
    test_normed = test_embeddings / test_norms
    prod_norms = np.linalg.norm(prod_embeddings, axis=1, keepdims=True) + 1e-10
    prod_normed = prod_embeddings / prod_norms

    # Nearest neighbor
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
    nn.fit(test_normed)
    distances, _ = nn.kneighbors(prod_normed)
    distances = distances.ravel()

    # Auto-threshold
    if distance_threshold is None:
        distance_threshold = float(np.mean(distances) + 1.5 * np.std(distances))
        logger.info(f"Auto-computed distance threshold: {distance_threshold:.4f}")

    # Find uncovered prompts
    uncovered_mask = distances > distance_threshold
    n_uncovered = int(np.sum(uncovered_mask))
    logger.info(f"{n_uncovered}/{len(distances)} production prompts are uncovered")

    # Initialize all as -1 (covered)
    gap_labels = np.full(len(distances), -1, dtype=int)

    if n_uncovered < min_gap_size:
        return gap_labels, distances, distance_threshold

    # Cluster the uncovered embeddings
    uncovered_embeddings = prod_embeddings[uncovered_mask]
    try:
        uncovered_cluster_labels = cluster_embeddings(
            uncovered_embeddings,
            method=clustering_method,
            min_cluster_size=min_gap_size,
        )
    except ClusteringError:
        # Clustering failed → treat all uncovered as one gap
        logger.warning(
            "Clustering of uncovered prompts failed. Treating all as one gap."
        )
        uncovered_cluster_labels = np.zeros(n_uncovered, dtype=int)

    # Map back to full array
    uncovered_indices = np.where(uncovered_mask)[0]
    for local_idx, global_idx in enumerate(uncovered_indices):
        gap_labels[global_idx] = uncovered_cluster_labels[local_idx]

    return gap_labels, distances, distance_threshold
```

### `coverage/__init__.py`

```python
from .detector import detect_coverage_gaps

__all__ = ["detect_coverage_gaps"]
```

---

## 2. `report.py` — Report Formatting

Dual formatting: rich tables if installed, plain text fallback.

```python
from __future__ import annotations

from .models import AnalysisReport, CoverageReport


def format_analysis_report(report: AnalysisReport) -> str:
    """Format AnalysisReport. Try rich, fall back to plain text."""
    try:
        return _format_analysis_rich(report)
    except ImportError:
        return _format_analysis_plain(report)


def format_coverage_report(report: CoverageReport) -> str:
    """Format CoverageReport. Try rich, fall back to plain text."""
    try:
        return _format_coverage_rich(report)
    except ImportError:
        return _format_coverage_plain(report)


def _format_analysis_plain(report: AnalysisReport) -> str:
    sep = "=" * 55
    thin = "-" * 55
    lines = [
        sep,
        "FAULTMAP ANALYSIS REPORT",
        sep,
        f"Total prompts:     {report.total_prompts}",
        f"Total failures:    {report.total_failures} ({report.baseline_failure_rate:.1%})",
        f"Scoring mode:      {report.scoring_mode}",
        f"Clustering:        {report.clustering_method}",
        f"Embedding model:   {report.embedding_model}",
        f"Significance:      alpha={report.significance_level}",
        f"Clusters tested:   {report.num_clusters_tested}",
        f"Significant:       {report.num_significant}",
    ]

    if not report.slices:
        lines.append(thin)
        lines.append("No statistically significant failure slices found.")
    else:
        for i, s in enumerate(report.slices, 1):
            lines.append(thin)
            lines.append(f'Slice {i}: "{s.name}"')
            lines.append(f"  Description:    {s.description}")
            lines.append(f"  Size:           {s.size} prompts")
            lines.append(f"  Failure rate:   {s.failure_rate:.1%} (vs {s.baseline_rate:.1%} baseline)")
            lines.append(f"  Effect size:    {s.effect_size:.1f}x")
            lines.append(f"  Adj. p-value:   {s.adjusted_p_value:.6f} ({s.test_used})")
            lines.append(f"  Examples:")
            for prompt in s.representative_prompts[:5]:
                truncated = prompt[:120] + ("..." if len(prompt) > 120 else "")
                lines.append(f"    - {truncated}")

    lines.append(sep)
    return "\n".join(lines)


def _format_analysis_rich(report: AnalysisReport) -> str:
    from rich.console import Console
    from rich.table import Table
    from io import StringIO

    console = Console(file=StringIO(), force_terminal=True, width=120)

    console.print("[bold]FAULTMAP ANALYSIS REPORT[/bold]", style="cyan")
    console.print(
        f"Prompts: {report.total_prompts} | "
        f"Failures: {report.total_failures} ({report.baseline_failure_rate:.1%}) | "
        f"Mode: {report.scoring_mode} | "
        f"Clustering: {report.clustering_method}"
    )
    console.print(
        f"Clusters tested: {report.num_clusters_tested} | "
        f"Significant: {report.num_significant} | "
        f"Alpha: {report.significance_level}"
    )

    if report.slices:
        table = Table(show_header=True, header_style="bold")
        table.add_column("#", width=3)
        table.add_column("Name", width=25)
        table.add_column("Size", width=6, justify="right")
        table.add_column("Fail Rate", width=10, justify="right")
        table.add_column("Baseline", width=10, justify="right")
        table.add_column("Effect", width=8, justify="right")
        table.add_column("Adj. p", width=10, justify="right")
        table.add_column("Test", width=6)

        for i, s in enumerate(report.slices, 1):
            table.add_row(
                str(i), s.name, str(s.size),
                f"{s.failure_rate:.1%}", f"{s.baseline_rate:.1%}",
                f"{s.effect_size:.1f}x", f"{s.adjusted_p_value:.4f}",
                s.test_used,
            )
        console.print(table)

        for i, s in enumerate(report.slices, 1):
            console.print(f"\n[bold]Slice {i}: {s.name}[/bold]")
            console.print(f"  {s.description}")
            console.print("  Examples:")
            for p in s.representative_prompts[:3]:
                truncated = p[:100] + ("..." if len(p) > 100 else "")
                console.print(f"    - {truncated}", style="dim")
    else:
        console.print("[green]No statistically significant failure slices found.[/green]")

    return console.file.getvalue()


def _format_coverage_plain(report: CoverageReport) -> str:
    sep = "=" * 55
    thin = "-" * 55
    lines = [
        sep,
        "FAULTMAP COVERAGE REPORT",
        sep,
        f"Test prompts:        {report.num_test_prompts}",
        f"Production prompts:  {report.num_production_prompts}",
        f"Coverage score:      {report.overall_coverage_score:.1%}",
        f"Distance threshold:  {report.distance_threshold:.4f}",
        f"Embedding model:     {report.embedding_model}",
        f"Gaps found:          {report.num_gaps}",
    ]

    if not report.gaps:
        lines.append(thin)
        lines.append("No significant coverage gaps found.")
    else:
        for i, g in enumerate(report.gaps, 1):
            lines.append(thin)
            lines.append(f'Gap {i}: "{g.name}"')
            lines.append(f"  Description:     {g.description}")
            lines.append(f"  Size:            {g.size} prompts")
            lines.append(f"  Mean distance:   {g.mean_distance:.4f}")
            lines.append(f"  Examples:")
            for prompt in g.representative_prompts[:5]:
                truncated = prompt[:120] + ("..." if len(prompt) > 120 else "")
                lines.append(f"    - {truncated}")

    lines.append(sep)
    return "\n".join(lines)


def _format_coverage_rich(report: CoverageReport) -> str:
    from rich.console import Console
    from rich.table import Table
    from io import StringIO

    console = Console(file=StringIO(), force_terminal=True, width=120)
    console.print("[bold]FAULTMAP COVERAGE REPORT[/bold]", style="cyan")
    console.print(
        f"Test: {report.num_test_prompts} | "
        f"Production: {report.num_production_prompts} | "
        f"Coverage: {report.overall_coverage_score:.1%}"
    )

    if report.gaps:
        table = Table(show_header=True, header_style="bold")
        table.add_column("#", width=3)
        table.add_column("Name", width=25)
        table.add_column("Size", width=6, justify="right")
        table.add_column("Mean Dist", width=10, justify="right")

        for i, g in enumerate(report.gaps, 1):
            table.add_row(str(i), g.name, str(g.size), f"{g.mean_distance:.4f}")
        console.print(table)
    else:
        console.print("[green]No significant coverage gaps found.[/green]")

    return console.file.getvalue()
```

---

## 3. `analyzer.py` — The Orchestrator

This is the main entry point. It wires scoring → embedding → clustering → testing → naming → reporting.

```python
from __future__ import annotations
import logging
import warnings
from typing import Optional

import numpy as np

from .embeddings import Embedder, get_embedder
from .exceptions import ConfigurationError
from .labeling import label_clusters
from .llm import AsyncLLMClient
from .models import (
    AnalysisReport, CoverageGap, CoverageReport,
    FailureSlice, ScoringResult,
)
from .report import format_analysis_report, format_coverage_report
from .utils import run_sync, validate_inputs

logger = logging.getLogger(__name__)


class SliceAnalyzer:
    """
    Main entry point for faultmap.

    Discovers failure slices and coverage gaps in LLM evaluations.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        embedding_model: str = "all-MiniLM-L6-v2",
        significance_level: float = 0.05,
        min_slice_size: int = 10,
        failure_threshold: float = 0.5,
        n_samples: int = 8,
        clustering_method: str = "hdbscan",
        max_concurrent_requests: int = 50,
        temperature: float = 1.0,
        consistency_threshold: float = 0.8,
    ) -> None:
        # Validate
        if clustering_method not in ("hdbscan", "agglomerative"):
            raise ConfigurationError(
                f"clustering_method must be 'hdbscan' or 'agglomerative', "
                f"got {clustering_method!r}"
            )
        if not 0 < significance_level < 1:
            raise ConfigurationError("significance_level must be in (0, 1)")
        if not 0 <= failure_threshold <= 1:
            raise ConfigurationError("failure_threshold must be in [0, 1]")
        if n_samples < 2:
            raise ConfigurationError("n_samples must be >= 2 for entropy scoring")

        self.model = model
        self.embedding_model = embedding_model
        self.significance_level = significance_level
        self.min_slice_size = min_slice_size
        self.failure_threshold = failure_threshold
        self.n_samples = n_samples
        self.clustering_method = clustering_method
        self.max_concurrent_requests = max_concurrent_requests
        self.temperature = temperature
        self.consistency_threshold = consistency_threshold

        self._embedder: Embedder = get_embedder(embedding_model)
        self._llm_client = AsyncLLMClient(
            model=model,
            max_concurrent_requests=max_concurrent_requests,
        )

    # ── analyze() ──────────────────────────────────────────

    def analyze(
        self,
        prompts: list[str],
        responses: list[str],
        scores: Optional[list[float]] = None,
        references: Optional[list[str]] = None,
    ) -> AnalysisReport:
        """
        Discover failure slices. Sync entry point.

        Mode detection:
        - scores → Mode 1 (precomputed)
        - references → Mode 2 (reference-based)
        - neither → Mode 3 (entropy, autonomous)
        - both → scores wins with warning
        """
        return run_sync(
            self._analyze_async(prompts, responses, scores, references)
        )

    async def _analyze_async(
        self,
        prompts: list[str],
        responses: list[str],
        scores: Optional[list[float]],
        references: Optional[list[str]],
    ) -> AnalysisReport:
        """
        Async orchestration pipeline:

        1.  VALIDATE inputs
        2.  DETECT scoring mode
        3.  SCORE → ScoringResult
        4.  BINARIZE → failures = score < threshold
        5.  Early return if 0 failures
        6.  EMBED prompts (NOT responses)
        7.  CLUSTER prompt embeddings
        8.  TEST each cluster (chi2 or Fisher)
        9.  BH CORRECT p-values
        10. FILTER → keep adjusted_p < alpha
        11. NAME significant clusters via LLM
        12. ASSEMBLE FailureSlice objects
        13. Return AnalysisReport
        """
        from .scoring import PrecomputedScorer, ReferenceScorer, EntropyScorer
        from .slicing import (
            cluster_embeddings, get_representative_prompts,
            test_cluster_failure_rate, benjamini_hochberg,
        )

        # 1. Validate
        validate_inputs(prompts, responses, scores, references)

        # 2. Mode detection
        if scores is not None and references is not None:
            warnings.warn(
                "Both scores and references provided. Using scores (Mode 1).",
                UserWarning, stacklevel=3,
            )
            references = None

        if scores is not None:
            scoring_mode = "precomputed"
            scorer = PrecomputedScorer(scores)
        elif references is not None:
            scoring_mode = "reference"
            scorer = ReferenceScorer(self._embedder, references)
        else:
            scoring_mode = "entropy"
            scorer = EntropyScorer(
                client=self._llm_client, embedder=self._embedder,
                n_samples=self.n_samples, temperature=self.temperature,
                consistency_threshold=self.consistency_threshold,
            )

        # 3. Score
        logger.info(f"Scoring mode: {scoring_mode}")
        scoring_result: ScoringResult = await scorer.score(prompts, responses)

        # 4. Binarize
        score_array = np.array(scoring_result.scores)
        failures = score_array < self.failure_threshold
        total_failures = int(np.sum(failures))
        total_prompts = len(prompts)
        baseline = total_failures / total_prompts if total_prompts > 0 else 0.0

        logger.info(
            f"Failures: {total_failures}/{total_prompts} ({baseline:.1%}) "
            f"at threshold={self.failure_threshold}"
        )

        # 5. Early return
        if total_failures == 0:
            return AnalysisReport(
                slices=[], total_prompts=total_prompts, total_failures=0,
                baseline_failure_rate=0.0, significance_level=self.significance_level,
                failure_threshold=self.failure_threshold, scoring_mode=scoring_mode,
                num_clusters_tested=0, num_significant=0,
                clustering_method=self.clustering_method,
                embedding_model=self.embedding_model,
            )

        # 6. Embed prompts
        logger.info("Embedding prompts...")
        prompt_embeddings = self._embedder.embed(prompts)

        # 7. Cluster
        logger.info(f"Clustering ({self.clustering_method})...")
        labels = cluster_embeddings(
            prompt_embeddings, method=self.clustering_method,
            min_cluster_size=self.min_slice_size,
        )

        unique_labels = sorted(set(labels))
        if -1 in unique_labels:
            unique_labels.remove(-1)

        # 8. Statistical testing
        logger.info(f"Testing {len(unique_labels)} clusters...")
        test_results = []
        for cid in unique_labels:
            mask = labels == cid
            result = test_cluster_failure_rate(
                cluster_failures=int(np.sum(failures[mask])),
                cluster_size=int(np.sum(mask)),
                total_failures=total_failures,
                total_size=total_prompts,
                cluster_id=cid,
            )
            test_results.append(result)

        # 9. BH correction
        corrected = benjamini_hochberg(test_results, alpha=self.significance_level)

        # 10. Filter
        significant = [
            r for r in corrected
            if r.adjusted_p_value < self.significance_level
        ]
        logger.info(
            f"{len(significant)}/{len(corrected)} clusters significant "
            f"at alpha={self.significance_level}"
        )

        if not significant:
            return AnalysisReport(
                slices=[], total_prompts=total_prompts,
                total_failures=total_failures, baseline_failure_rate=baseline,
                significance_level=self.significance_level,
                failure_threshold=self.failure_threshold, scoring_mode=scoring_mode,
                num_clusters_tested=len(corrected), num_significant=0,
                clustering_method=self.clustering_method,
                embedding_model=self.embedding_model,
            )

        # 11. Name
        clusters_texts = []
        clusters_all_indices = []
        for r in significant:
            rep_prompts, _ = get_representative_prompts(
                prompt_embeddings, labels, r.cluster_id, prompts, top_k=10
            )
            clusters_texts.append(rep_prompts)
            # ALL indices in this cluster
            all_idx = np.where(labels == r.cluster_id)[0].tolist()
            clusters_all_indices.append(all_idx)

        logger.info(f"Naming {len(significant)} clusters...")
        cluster_labels = await label_clusters(
            self._llm_client, clusters_texts, context="failure slice"
        )

        # 12. Assemble
        slices: list[FailureSlice] = []
        for r, label, rep_texts, all_idx in zip(
            significant, cluster_labels, clusters_texts, clusters_all_indices
        ):
            # Build examples: top-5 with prompt, response, score
            examples = []
            for idx in all_idx[:5]:
                examples.append({
                    "prompt": prompts[idx],
                    "response": responses[idx],
                    "score": float(score_array[idx]),
                })

            effect = r.failure_rate / baseline if baseline > 0 else float('inf')

            slices.append(FailureSlice(
                name=label.name,
                description=label.description,
                size=r.size,
                failure_rate=r.failure_rate,
                baseline_rate=baseline,
                effect_size=round(effect, 2),
                p_value=r.p_value,
                adjusted_p_value=r.adjusted_p_value,
                test_used=r.test_used,
                sample_indices=all_idx,
                examples=examples,
                representative_prompts=rep_texts[:5],
                cluster_id=r.cluster_id,
            ))

        report = AnalysisReport(
            slices=slices, total_prompts=total_prompts,
            total_failures=total_failures, baseline_failure_rate=baseline,
            significance_level=self.significance_level,
            failure_threshold=self.failure_threshold, scoring_mode=scoring_mode,
            num_clusters_tested=len(corrected), num_significant=len(slices),
            clustering_method=self.clustering_method,
            embedding_model=self.embedding_model,
            metadata={"scoring_metadata": scoring_result.metadata},
        )

        logger.info(report.summary())
        return report

    # ── audit_coverage() ───────────────────────────────────

    def audit_coverage(
        self,
        test_prompts: list[str],
        production_prompts: list[str],
        distance_threshold: float | None = None,
        min_gap_size: int = 5,
    ) -> CoverageReport:
        """Sync entry point for coverage auditing."""
        return run_sync(self._audit_coverage_async(
            test_prompts, production_prompts, distance_threshold, min_gap_size
        ))

    async def _audit_coverage_async(
        self,
        test_prompts: list[str],
        production_prompts: list[str],
        distance_threshold: float | None,
        min_gap_size: int,
    ) -> CoverageReport:
        """
        Pipeline:
        1. Validate
        2. Embed test + production prompts
        3. Detect gaps (NN distances + clustering)
        4. Get representatives per gap
        5. Name gaps via LLM
        6. Assemble CoverageReport
        """
        from .coverage import detect_coverage_gaps
        from .slicing.clustering import get_representative_prompts

        if not test_prompts:
            raise ConfigurationError("test_prompts must be non-empty")
        if not production_prompts:
            raise ConfigurationError("production_prompts must be non-empty")

        # Embed
        logger.info("Embedding test prompts...")
        test_emb = self._embedder.embed(test_prompts)
        logger.info("Embedding production prompts...")
        prod_emb = self._embedder.embed(production_prompts)

        # Detect gaps
        gap_labels, nn_distances, used_threshold = detect_coverage_gaps(
            test_embeddings=test_emb, prod_embeddings=prod_emb,
            prod_prompts=production_prompts,
            distance_threshold=distance_threshold,
            min_gap_size=min_gap_size,
            clustering_method=self.clustering_method,
        )

        unique_gaps = sorted(set(gap_labels))
        if -1 in unique_gaps:
            unique_gaps.remove(-1)

        if not unique_gaps:
            total = len(production_prompts)
            uncovered = int(np.sum(gap_labels != -1))
            return CoverageReport(
                gaps=[], num_test_prompts=len(test_prompts),
                num_production_prompts=total, num_gaps=0,
                overall_coverage_score=1.0 - (uncovered / total if total else 0),
                distance_threshold=used_threshold,
                embedding_model=self.embedding_model,
            )

        # Representatives + metadata per gap
        clusters_texts = []
        gaps_meta = []
        for cid in unique_gaps:
            mask = gap_labels == cid
            size = int(np.sum(mask))
            mean_dist = float(np.mean(nn_distances[mask]))
            rep_prompts, rep_indices = get_representative_prompts(
                prod_emb, gap_labels, cid, production_prompts, top_k=10
            )
            clusters_texts.append(rep_prompts)
            all_idx = np.where(mask)[0].tolist()
            gaps_meta.append((cid, size, mean_dist, rep_prompts[:5], all_idx))

        # Name gaps
        logger.info(f"Naming {len(unique_gaps)} coverage gaps...")
        cluster_labels = await label_clusters(
            self._llm_client, clusters_texts, context="coverage gap"
        )

        # Assemble
        total_uncovered = int(np.sum(gap_labels >= 0))
        total = len(production_prompts)

        gaps: list[CoverageGap] = []
        for label, (cid, size, mean_dist, rep_prompts, all_idx) in zip(
            cluster_labels, gaps_meta
        ):
            gaps.append(CoverageGap(
                name=label.name, description=label.description,
                size=size, mean_distance=mean_dist,
                representative_prompts=rep_prompts,
                prompt_indices=all_idx, cluster_id=cid,
            ))

        gaps.sort(key=lambda g: g.mean_distance, reverse=True)

        report = CoverageReport(
            gaps=gaps, num_test_prompts=len(test_prompts),
            num_production_prompts=total, num_gaps=len(gaps),
            overall_coverage_score=1.0 - (total_uncovered / total if total else 0),
            distance_threshold=used_threshold,
            embedding_model=self.embedding_model,
        )

        logger.info(report.summary())
        return report
```

---

## 4. Day 5 Tests

### `tests/test_coverage/test_detector.py`

```python
import numpy as np
import pytest
from faultmap.coverage.detector import detect_coverage_gaps
from faultmap.exceptions import ConfigurationError


def _make_coverage_data(seed=42):
    """
    Test embeddings cluster in region A.
    Production embeddings cluster in region A (covered) + region B (gap).
    """
    rng = np.random.default_rng(seed)
    dim = 32

    # Region A: test + some production
    center_a = rng.standard_normal(dim) * 5
    test_embs = center_a + rng.standard_normal((50, dim)) * 0.2
    prod_covered = center_a + rng.standard_normal((30, dim)) * 0.2

    # Region B: only production (gap)
    center_b = -center_a  # far away
    prod_gap = center_b + rng.standard_normal((20, dim)) * 0.2

    prod_embs = np.vstack([prod_covered, prod_gap])
    prod_prompts = [f"prompt-{i}" for i in range(50)]

    return test_embs.astype(np.float32), prod_embs.astype(np.float32), prod_prompts


class TestDetectCoverageGaps:
    def test_finds_gap(self):
        test_embs, prod_embs, prompts = _make_coverage_data()
        gap_labels, distances, threshold = detect_coverage_gaps(
            test_embs, prod_embs, prompts, min_gap_size=5,
        )
        # Should find gap in region B (indices 30-49)
        has_gap = np.any(gap_labels[30:] >= 0)
        assert has_gap, "Should detect a gap in the uncovered region"

    def test_all_covered(self):
        rng = np.random.default_rng(42)
        dim = 16
        embs = rng.standard_normal((50, dim)).astype(np.float32)
        gap_labels, _, _ = detect_coverage_gaps(
            embs, embs, [f"p-{i}" for i in range(50)],
            min_gap_size=5,
        )
        # All should be -1 (covered) since test == prod
        assert np.all(gap_labels == -1)

    def test_empty_test_raises(self):
        with pytest.raises(ConfigurationError):
            detect_coverage_gaps(
                np.empty((0, 16)), np.random.randn(10, 16).astype(np.float32),
                [f"p-{i}" for i in range(10)],
            )

    def test_empty_prod_returns_empty(self):
        gap_labels, distances, _ = detect_coverage_gaps(
            np.random.randn(10, 16).astype(np.float32),
            np.empty((0, 16), dtype=np.float32),
            [],
        )
        assert len(gap_labels) == 0
```

### `tests/test_report.py`

```python
from faultmap.models import AnalysisReport, FailureSlice, CoverageReport
from faultmap.report import format_analysis_report, format_coverage_report


class TestFormatAnalysisReport:
    def test_empty_slices(self):
        report = AnalysisReport(
            slices=[], total_prompts=100, total_failures=15,
            baseline_failure_rate=0.15, significance_level=0.05,
            failure_threshold=0.5, scoring_mode="precomputed",
            num_clusters_tested=5, num_significant=0,
            clustering_method="hdbscan", embedding_model="test",
        )
        text = format_analysis_report(report)
        assert "No statistically significant" in text
        assert "100" in text

    def test_with_slices(self):
        s = FailureSlice(
            name="Legal Questions", description="About legal topics",
            size=20, failure_rate=0.8, baseline_rate=0.15,
            effect_size=5.3, p_value=0.001, adjusted_p_value=0.005,
            test_used="chi2", sample_indices=list(range(20)),
            examples=[{"prompt": "p", "response": "r", "score": 0.1}],
            representative_prompts=["How do I comply with X?"],
            cluster_id=0,
        )
        report = AnalysisReport(
            slices=[s], total_prompts=100, total_failures=15,
            baseline_failure_rate=0.15, significance_level=0.05,
            failure_threshold=0.5, scoring_mode="precomputed",
            num_clusters_tested=5, num_significant=1,
            clustering_method="hdbscan", embedding_model="test",
        )
        text = format_analysis_report(report)
        assert "Legal Questions" in text
        assert "80.0%" in text
        assert "5.3x" in text


class TestFormatCoverageReport:
    def test_no_gaps(self):
        report = CoverageReport(
            gaps=[], num_test_prompts=50, num_production_prompts=100,
            num_gaps=0, overall_coverage_score=1.0,
            distance_threshold=0.5, embedding_model="test",
        )
        text = format_coverage_report(report)
        assert "No significant coverage gaps" in text
```

### `tests/test_analyzer.py` — Integration

```python
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from faultmap.analyzer import SliceAnalyzer
from faultmap.exceptions import ConfigurationError


@pytest.fixture
def mock_embedder():
    """Embedder that creates distinct clusters based on prompt content."""
    embedder = MagicMock()
    embedder.model_name = "mock"

    def mock_embed(texts):
        dim = 64
        rng = np.random.default_rng(42)
        embs = []
        for t in texts:
            # Hash-based deterministic embedding
            seed = hash(t) % (2**31)
            local_rng = np.random.default_rng(seed)
            embs.append(local_rng.standard_normal(dim))
        return np.array(embs, dtype=np.float32)

    embedder.embed = mock_embed
    embedder.dimension = 64
    return embedder


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


class TestAnalyzeMode1:
    def test_mode_detection(self, mock_embedder):
        """Providing scores → Mode 1 (precomputed)."""
        analyzer = SliceAnalyzer.__new__(SliceAnalyzer)
        analyzer.significance_level = 0.05
        analyzer.min_slice_size = 5
        analyzer.failure_threshold = 0.5
        analyzer.clustering_method = "hdbscan"
        analyzer.embedding_model = "mock"
        analyzer._embedder = mock_embedder
        analyzer._llm_client = AsyncMock()
        analyzer._llm_client.complete.return_value = (
            "Name: Test Slice\nDescription: Test description"
        )
        analyzer.n_samples = 8
        analyzer.temperature = 1.0
        analyzer.consistency_threshold = 0.8

        n = 100
        prompts = [f"prompt-{i}" for i in range(n)]
        responses = [f"response-{i}" for i in range(n)]
        # All pass (scores > threshold)
        scores = [0.9] * n

        report = analyzer.analyze(prompts, responses, scores=scores)
        assert report.scoring_mode == "precomputed"
        assert report.total_failures == 0
        assert len(report.slices) == 0

    def test_both_scores_and_references_warns(self, mock_embedder):
        """Both provided → scores wins with warning."""
        analyzer = SliceAnalyzer.__new__(SliceAnalyzer)
        analyzer.significance_level = 0.05
        analyzer.min_slice_size = 5
        analyzer.failure_threshold = 0.5
        analyzer.clustering_method = "hdbscan"
        analyzer.embedding_model = "mock"
        analyzer._embedder = mock_embedder
        analyzer._llm_client = AsyncMock()
        analyzer.n_samples = 8
        analyzer.temperature = 1.0
        analyzer.consistency_threshold = 0.8

        with pytest.warns(UserWarning, match="Both scores and references"):
            report = analyzer.analyze(
                ["a"], ["b"], scores=[0.9], references=["c"]
            )
        assert report.scoring_mode == "precomputed"
```

---

## Verification

After completing Day 5:
```bash
# Run all tests
pytest tests/ -v

# Verify the package works end-to-end
python -c "
from faultmap import SliceAnalyzer
analyzer = SliceAnalyzer(model='gpt-4o-mini')
print('SliceAnalyzer created successfully')
print(f'Embedding model: {analyzer.embedding_model}')
print(f'Clustering: {analyzer.clustering_method}')
"
```
