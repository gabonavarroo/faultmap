from __future__ import annotations

import logging
import warnings

import numpy as np

from .embeddings import Embedder, get_embedder
from .exceptions import ConfigurationError
from .labeling import label_clusters
from .llm import AsyncLLMClient
from .models import (
    AnalysisReport,
    ComparisonReport,
    CoverageGap,
    CoverageReport,
    FailureSlice,
    SliceComparison,
    ScoringResult,
)
from .utils import run_sync, validate_comparison_inputs, validate_inputs

logger = logging.getLogger(__name__)


class SliceAnalyzer:
    """Discover failure slices and coverage gaps in LLM evaluations.

    ``SliceAnalyzer`` is the main entry point for faultmap. It provides two
    methods:

    - :meth:`analyze` — find input slices with statistically elevated failure rates
    - :meth:`audit_coverage` — find semantic blind spots in a test suite

    Example::

        from faultmap import SliceAnalyzer

        analyzer = SliceAnalyzer(model="gpt-4o-mini")
        report = analyzer.analyze(prompts, responses, scores=scores)
        print(report)

        coverage = analyzer.audit_coverage(test_prompts, prod_prompts)
        print(coverage)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        embedding_max_text_chars: int | None = 2000,
        embedding_request_kwargs: dict[str, object] | None = None,
        embedding_usage_kwargs: dict[str, dict[str, object]] | None = None,
        significance_level: float = 0.05,
        min_slice_size: int = 10,
        failure_threshold: float = 0.5,
        n_samples: int = 8,
        clustering_method: str = "hdbscan",
        max_concurrent_requests: int = 50,
        temperature: float = 1.0,
        consistency_threshold: float = 0.8,
    ) -> None:
        """Initialize a SliceAnalyzer.

        Args:
            model: litellm model string used for LLM calls (cluster naming and
                Mode 3 response sampling). Supports 100+ providers, e.g.
                ``"gpt-4o-mini"``, ``"anthropic/claude-3-haiku-20240307"``,
                ``"ollama/mistral"``.
            embedding_model: Embedding model name. Local sentence-transformers models
                are auto-detected by prefix (``"all-MiniLM-"``, ``"all-mpnet-"``,
                ``"paraphrase-"``); all others route to ``APIEmbedder`` via litellm.
                The default uses an API-backed embedding model so
                ``pip install faultmap`` works without optional extras. Local models
                require ``pip install faultmap[local]``.
            embedding_max_text_chars: Character-level truncation applied to API
                embedding inputs before sending requests. Helps avoid provider token
                limits on long texts. ``None`` disables truncation. Ignored for local
                models. Default ``2000``.
            embedding_request_kwargs: Optional kwargs passed to API embedding calls
                for all usages. Ignored for local models.
            embedding_usage_kwargs: Optional per-usage API kwargs keyed by
                ``"query"`` and/or ``"document"`` for asymmetric embedding models.
                Ignored for local models.
            significance_level: FDR alpha for Benjamini-Hochberg correction. Slices
                with ``adjusted_p_value < significance_level`` are reported.
                Default ``0.05``.
            min_slice_size: Minimum number of prompts a cluster must contain to be
                tested. Smaller clusters are silently discarded. Default ``10``.
            failure_threshold: Score cutoff for binary pass/fail. A prompt with
                ``score < failure_threshold`` is counted as a failure. Default ``0.5``.
            n_samples: Mode 3 only. Number of additional LLM responses sampled per
                prompt for entropy estimation. Must be >= 2. Higher values give more
                accurate entropy estimates at greater API cost. Default ``8``.
            clustering_method: Clustering algorithm.
                - ``"hdbscan"`` (default): automatically discovers the number of
                  clusters using density-based clustering (sklearn >= 1.3 built-in).
                - ``"agglomerative"``: Ward linkage with silhouette-based k-selection
                  over ``[5, 10, 15, 20, 25, 30]``. More predictable cluster count.
            max_concurrent_requests: Maximum number of parallel LLM API calls.
                Controlled via asyncio semaphore. Reduce if hitting rate limits.
                Default ``50``.
            temperature: Mode 3 only. Sampling temperature for response diversity.
                Higher values increase entropy estimation accuracy. Default ``1.0``.
            consistency_threshold: Mode 3 only. Cosine similarity threshold above
                which a sampled response is considered "consistent" with the original.
                Default ``0.8``.

        Raises:
            ConfigurationError: If ``clustering_method`` is not ``"hdbscan"`` or
                ``"agglomerative"``, ``significance_level`` is not in (0, 1),
                ``failure_threshold`` is not in [0, 1], ``n_samples < 2``,
                ``min_slice_size <= 0``, ``max_concurrent_requests <= 0``,
                ``temperature < 0``, or ``consistency_threshold`` is not in [0, 1].
        """
        # Validate
        if clustering_method not in ("hdbscan", "agglomerative"):
            raise ConfigurationError(
                f"clustering_method must be 'hdbscan' or 'agglomerative', "
                f"got {clustering_method!r}"
            )
        if not 0 < significance_level < 1:
            raise ConfigurationError("significance_level must be in (0, 1)")
        if min_slice_size <= 0:
            raise ConfigurationError("min_slice_size must be > 0")
        if not 0 <= failure_threshold <= 1:
            raise ConfigurationError("failure_threshold must be in [0, 1]")
        if n_samples < 2:
            raise ConfigurationError("n_samples must be >= 2 for entropy scoring")
        if max_concurrent_requests <= 0:
            raise ConfigurationError("max_concurrent_requests must be > 0")
        if temperature < 0:
            raise ConfigurationError("temperature must be >= 0")
        if not 0 <= consistency_threshold <= 1:
            raise ConfigurationError("consistency_threshold must be in [0, 1]")

        self.model = model
        self.embedding_model = embedding_model
        self.embedding_max_text_chars = embedding_max_text_chars
        self.embedding_request_kwargs = embedding_request_kwargs
        self.embedding_usage_kwargs = embedding_usage_kwargs
        self.significance_level = significance_level
        self.min_slice_size = min_slice_size
        self.failure_threshold = failure_threshold
        self.n_samples = n_samples
        self.clustering_method = clustering_method
        self.max_concurrent_requests = max_concurrent_requests
        self.temperature = temperature
        self.consistency_threshold = consistency_threshold

        self._embedder: Embedder = get_embedder(
            embedding_model,
            api_max_text_chars=embedding_max_text_chars,
            api_request_kwargs=embedding_request_kwargs,
            api_usage_request_kwargs=embedding_usage_kwargs,
        )
        self._llm_client = AsyncLLMClient(
            model=model,
            max_concurrent_requests=max_concurrent_requests,
        )

    # ── analyze() ──────────────────────────────────────────

    def analyze(
        self,
        prompts: list[str],
        responses: list[str],
        scores: list[float] | None = None,
        references: list[str] | None = None,
    ) -> AnalysisReport:
        """Discover failure slices — input regions where your LLM fails significantly more.

        Embeds prompts, clusters them, runs statistical tests on each cluster, applies
        Benjamini-Hochberg FDR correction, and names significant clusters via LLM.

        Scoring mode is auto-detected from the arguments:

        - **Mode 1** (``scores`` provided): use pre-computed scores directly.
        - **Mode 2** (``references`` provided): score by cosine similarity between
          response and reference embeddings.
        - **Mode 3** (neither): autonomous scoring via semantic entropy and
          self-consistency (makes additional LLM API calls).
        - Both provided: Mode 1 wins, Mode 2 is ignored (``UserWarning`` raised).

        Args:
            prompts: Input prompts to analyze. Must be non-empty and the same length
                as ``responses``.
            responses: Model responses corresponding to each prompt.
            scores: Mode 1. Pre-computed quality scores in [0, 1] where higher is
                better. Must be the same length as ``prompts``.
            references: Mode 2. Ground-truth reference answers. Must be the same
                length as ``prompts``.

        Returns:
            :class:`AnalysisReport` with ``slices`` sorted by adjusted p-value
            ascending (most significant first). ``print(report)`` produces formatted
            output. ``report.to_dict()`` returns a JSON-serializable dict.

        Raises:
            ConfigurationError: If inputs are empty or length-mismatched.
        """
        return run_sync(
            self._analyze_async(prompts, responses, scores, references)
        )

    async def _analyze_async(
        self,
        prompts: list[str],
        responses: list[str],
        scores: list[float] | None,
        references: list[str] | None,
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
        from .scoring import EntropyScorer, PrecomputedScorer, ReferenceScorer
        from .slicing import (
            benjamini_hochberg,
            cluster_embeddings,
            get_representative_prompts,
            test_cluster_failure_rate,
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
                metadata={"scoring_metadata": scoring_result.metadata},
            )

        # 6. Embed prompts
        logger.info("Embedding prompts...")
        prompt_embeddings = self._embedder.embed_queries(prompts)

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
                metadata={"scoring_metadata": scoring_result.metadata},
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
        """Find semantic blind spots in a test suite by comparing against production traffic.

        Embeds both test and production prompts, uses k-nearest-neighbors to find
        production prompts that have no semantically similar test prompt, clusters those
        uncovered prompts into gap clusters, and names each gap via LLM.

        Args:
            test_prompts: Prompts from your evaluation / test suite.
            production_prompts: Prompts from real production traffic (e.g. application logs).
            distance_threshold: L2 distance cutoff in embedding space. Production prompts
                farther than this from any test prompt are considered "uncovered".
                If ``None`` (default), auto-computed as ``mean(distances) + 1.5 * std(distances)``.
                Set explicitly if the auto-threshold behaves unexpectedly on your data.
            min_gap_size: Minimum number of production prompts required to form a
                reportable gap cluster. Smaller clusters are discarded. Default ``5``.

        Returns:
            :class:`CoverageReport` with ``gaps`` sorted by ``mean_distance`` descending
            (most severe gaps first). ``print(coverage)`` produces formatted output.
            ``coverage.to_dict()`` returns a JSON-serializable dict.

        Raises:
            ConfigurationError: If ``test_prompts`` or ``production_prompts`` is
                empty, ``min_gap_size <= 0``, or ``distance_threshold < 0``.
        """
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
        if min_gap_size <= 0:
            raise ConfigurationError("min_gap_size must be > 0")
        if distance_threshold is not None and distance_threshold < 0:
            raise ConfigurationError("distance_threshold must be >= 0")

        # Embed
        logger.info("Embedding test prompts...")
        test_emb = self._embedder.embed_queries(test_prompts)
        logger.info("Embedding production prompts...")
        prod_emb = self._embedder.embed_queries(production_prompts)

        # Detect gaps
        gap_labels, nn_distances, used_threshold = detect_coverage_gaps(
            test_embeddings=test_emb, prod_embeddings=prod_emb,
            prod_prompts=production_prompts,
            distance_threshold=distance_threshold,
            min_gap_size=min_gap_size,
            clustering_method=self.clustering_method,
        )

        total = len(production_prompts)
        total_uncovered = int(np.sum(gap_labels != -1))
        unclustered_prompt_indices = np.where(gap_labels == -2)[0].tolist()
        coverage_metadata = {
            "num_uncovered_total": total_uncovered,
            "num_clustered_uncovered": int(np.sum(gap_labels >= 0)),
            "num_unclustered_uncovered": len(unclustered_prompt_indices),
            "unclustered_prompt_indices": unclustered_prompt_indices,
        }

        unique_gaps = sorted(set(gap_labels))
        if -1 in unique_gaps:
            unique_gaps.remove(-1)
        if -2 in unique_gaps:
            unique_gaps.remove(-2)

        if not unique_gaps:
            return CoverageReport(
                gaps=[], num_test_prompts=len(test_prompts),
                num_production_prompts=total, num_gaps=0,
                overall_coverage_score=1.0 - (total_uncovered / total if total else 0),
                distance_threshold=used_threshold,
                embedding_model=self.embedding_model,
                metadata=coverage_metadata,
            )

        # Representatives + metadata per gap
        clusters_texts = []
        gaps_meta = []
        for cid in unique_gaps:
            mask = gap_labels == cid
            size = int(np.sum(mask))
            mean_dist = float(np.mean(nn_distances[mask]))
            rep_prompts, _ = get_representative_prompts(
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
            metadata=coverage_metadata,
        )

        logger.info(report.summary())
        return report

    # ── compare_models() ───────────────────────────────────

    def compare_models(
        self,
        prompts: list[str],
        responses_a: list[str],
        responses_b: list[str],
        *,
        scores_a: list[float] | None = None,
        scores_b: list[float] | None = None,
        references: list[str] | None = None,
        model_a_name: str = "Model A",
        model_b_name: str = "Model B",
    ) -> ComparisonReport:
        """Compare two models on the same prompt set using McNemar's test.

        Embeds prompts once (shared between both models), clusters them, and runs
        per-slice McNemar's test to find semantic regions where one model is
        statistically better. Applies Benjamini-Hochberg FDR correction across slices
        and names significant slices via LLM.

        Scoring mode is auto-detected from arguments — identical to :meth:`analyze`:

        - **Mode 1** (``scores_a`` + ``scores_b`` provided): use pre-computed scores.
          Both must be provided or both must be ``None``; asymmetric raises
          :class:`~faultmap.exceptions.ConfigurationError`.
        - **Mode 2** (``references`` provided, no scores): score each model's responses
          against shared references via cosine similarity (``ReferenceScorer``).
        - **Mode 3** (neither): autonomous scoring via semantic entropy and
          self-consistency. Runs ``EntropyScorer`` independently on each model —
          **2× the API calls** of a single :meth:`analyze` call.
        - Both ``scores`` and ``references`` provided: Mode 1 wins with a
          ``UserWarning``.

        Args:
            prompts: Shared input prompts — identical for both models. Must be
                non-empty and equal in length to ``responses_a`` and ``responses_b``.
            responses_a: Responses from Model A, one per prompt.
            responses_b: Responses from Model B, one per prompt.
            scores_a: Mode 1. Pre-computed quality scores for Model A in ``[0, 1]``.
                Must have the same length as ``prompts``. Required when ``scores_b``
                is provided.
            scores_b: Mode 1. Pre-computed quality scores for Model B in ``[0, 1]``.
                Must have the same length as ``prompts``. Required when ``scores_a``
                is provided.
            references: Mode 2. Ground-truth reference answers shared by both models.
                Must have the same length as ``prompts``.
            model_a_name: Display name for Model A used in the report.
                Default ``"Model A"``.
            model_b_name: Display name for Model B used in the report.
                Default ``"Model B"``.

        Returns:
            :class:`ComparisonReport` with ``slices`` sorted by adjusted p-value
            ascending (most significant first). ``print(comparison)`` produces
            formatted output. ``comparison.to_dict()`` returns a JSON-serializable dict.

        Raises:
            ConfigurationError: If inputs are empty, length-mismatched, or scores are
                provided asymmetrically (``scores_a`` without ``scores_b`` or vice versa).
        """
        return run_sync(
            self._compare_models_async(
                prompts, responses_a, responses_b,
                scores_a=scores_a, scores_b=scores_b,
                references=references,
                model_a_name=model_a_name, model_b_name=model_b_name,
            )
        )

    async def _compare_models_async(
        self,
        prompts: list[str],
        responses_a: list[str],
        responses_b: list[str],
        *,
        scores_a: list[float] | None,
        scores_b: list[float] | None,
        references: list[str] | None,
        model_a_name: str,
        model_b_name: str,
    ) -> ComparisonReport:
        """
        14-step model comparison pipeline:

         1. VALIDATE       validate_comparison_inputs()
         2. MODE DETECT    scores_a+b → Mode 1 | references → Mode 2 | neither → Mode 3
         3. SCORE A        scorer_a.score(prompts, responses_a) → ScoringResult
         4. SCORE B        scorer_b.score(prompts, responses_b) → ScoringResult
         5. BINARIZE       failures_a/b = score_array < failure_threshold
         6. GLOBAL TEST    test_mcnemar(global b, global c) on all prompts
         7. EMBED          embedder.embed_queries(prompts) [once, shared]
         8. CLUSTER        cluster_embeddings() → labels (N,)
         9. PER-SLICE TEST test_mcnemar per cluster
        10. BH CORRECT     benjamini_hochberg_comparison()
        11. FILTER         keep adjusted_p < significance_level
        12. NAME           label_clusters() for significant slices
        13. ASSEMBLE       SliceComparison objects
        14. RETURN         ComparisonReport
        """
        from .comparison import benjamini_hochberg_comparison, test_mcnemar
        from .scoring import EntropyScorer, PrecomputedScorer, ReferenceScorer
        from .slicing import cluster_embeddings, get_representative_prompts

        # 1. Validate
        validate_comparison_inputs(
            prompts, responses_a, responses_b, scores_a, scores_b, references
        )

        # 2. Mode detection
        if scores_a is not None and references is not None:
            warnings.warn(
                "Both scores and references provided. Using scores (Mode 1).",
                UserWarning, stacklevel=3,
            )
            references = None

        if scores_a is not None:
            scoring_mode = "precomputed"
            scorer_a = PrecomputedScorer(scores_a)
            scorer_b = PrecomputedScorer(scores_b)  # type: ignore[arg-type]
        elif references is not None:
            scoring_mode = "reference"
            scorer_a = ReferenceScorer(self._embedder, references)
            scorer_b = ReferenceScorer(self._embedder, references)
        else:
            scoring_mode = "entropy"
            scorer_a = EntropyScorer(
                client=self._llm_client, embedder=self._embedder,
                n_samples=self.n_samples, temperature=self.temperature,
                consistency_threshold=self.consistency_threshold,
            )
            scorer_b = EntropyScorer(
                client=self._llm_client, embedder=self._embedder,
                n_samples=self.n_samples, temperature=self.temperature,
                consistency_threshold=self.consistency_threshold,
            )

        # 3. Score A
        logger.info(f"Scoring Model A ({scoring_mode})...")
        scoring_result_a: ScoringResult = await scorer_a.score(prompts, responses_a)

        # 4. Score B
        logger.info(f"Scoring Model B ({scoring_mode})...")
        scoring_result_b: ScoringResult = await scorer_b.score(prompts, responses_b)

        # 5. Binarize
        score_array_a = np.array(scoring_result_a.scores)
        score_array_b = np.array(scoring_result_b.scores)
        failures_a = score_array_a < self.failure_threshold
        failures_b = score_array_b < self.failure_threshold

        total_prompts = len(prompts)
        failure_rate_a = float(np.mean(failures_a))
        failure_rate_b = float(np.mean(failures_b))

        logger.info(
            f"Failure rates: A={failure_rate_a:.1%}, B={failure_rate_b:.1%} "
            f"at threshold={self.failure_threshold}"
        )

        # 6. Global McNemar test (single test across all prompts, no BH correction)
        # b = A passes AND B fails ("A wins"); c = A fails AND B passes ("B wins")
        b_global = int(np.sum(~failures_a & failures_b))
        c_global = int(np.sum(failures_a & ~failures_b))

        global_result = test_mcnemar(b_global, c_global, cluster_id=-1, size=total_prompts)
        global_winner = (
            global_result.winner
            if global_result.p_value < self.significance_level
            else "tie"
        )

        # 7. Embed prompts (once, shared between both models)
        logger.info("Embedding prompts...")
        prompt_embeddings = self._embedder.embed_queries(prompts)

        # 8. Cluster
        logger.info(f"Clustering ({self.clustering_method})...")
        labels = cluster_embeddings(
            prompt_embeddings, method=self.clustering_method,
            min_cluster_size=self.min_slice_size,
        )

        unique_labels = sorted(set(labels))
        if -1 in unique_labels:
            unique_labels.remove(-1)

        # 9. Per-slice McNemar test
        logger.info(f"Testing {len(unique_labels)} clusters...")
        slice_results = []
        for cid in unique_labels:
            mask = labels == cid
            b_slice = int(np.sum(~failures_a[mask] & failures_b[mask]))
            c_slice = int(np.sum(failures_a[mask] & ~failures_b[mask]))
            result = test_mcnemar(
                b_slice, c_slice, cluster_id=cid, size=int(np.sum(mask))
            )
            slice_results.append(result)

        # 10. BH correction
        corrected = benjamini_hochberg_comparison(
            slice_results, alpha=self.significance_level
        )

        # 11. Filter significant slices
        significant = [
            r for r in corrected
            if r.adjusted_p_value < self.significance_level
        ]
        logger.info(
            f"{len(significant)}/{len(corrected)} slices significant "
            f"at alpha={self.significance_level}"
        )

        if not significant:
            return ComparisonReport(
                slices=[],
                total_prompts=total_prompts,
                model_a_name=model_a_name,
                model_b_name=model_b_name,
                failure_rate_a=failure_rate_a,
                failure_rate_b=failure_rate_b,
                global_p_value=global_result.p_value,
                global_test_used=global_result.test_used,
                global_winner=global_winner,
                global_advantage_rate=global_result.advantage_rate,
                significance_level=self.significance_level,
                failure_threshold=self.failure_threshold,
                scoring_mode=scoring_mode,
                num_clusters_tested=len(corrected),
                num_significant=0,
                clustering_method=self.clustering_method,
                embedding_model=self.embedding_model,
            )

        # 12. Name significant slices
        clusters_texts = []
        clusters_all_indices = []
        for r in significant:
            rep_prompts, _ = get_representative_prompts(
                prompt_embeddings, labels, r.cluster_id, prompts, top_k=10
            )
            clusters_texts.append(rep_prompts)
            all_idx = np.where(labels == r.cluster_id)[0].tolist()
            clusters_all_indices.append(all_idx)

        logger.info(f"Naming {len(significant)} comparison slices...")
        cluster_labels = await label_clusters(
            self._llm_client, clusters_texts, context="model comparison slice"
        )

        # 13. Assemble SliceComparison objects
        slices: list[SliceComparison] = []
        for r, label, rep_texts, all_idx in zip(
            significant, cluster_labels, clusters_texts, clusters_all_indices
        ):
            cluster_mask = labels == r.cluster_id

            fr_a = float(np.mean(failures_a[cluster_mask]))
            fr_b = float(np.mean(failures_b[cluster_mask]))
            concordant_pass = int(
                np.sum(~failures_a[cluster_mask] & ~failures_b[cluster_mask])
            )
            concordant_fail = int(
                np.sum(failures_a[cluster_mask] & failures_b[cluster_mask])
            )

            # All items in `significant` already have adj_p < significance_level;
            # winner is directional (based on advantage_rate) from test_mcnemar.
            winner = r.winner

            # Top-5 discordant examples (prompts where the two models disagree)
            examples = []
            for idx in all_idx:
                if len(examples) >= 5:
                    break
                if bool(failures_a[idx]) != bool(failures_b[idx]):
                    examples.append({
                        "prompt": prompts[idx],
                        "response_a": responses_a[idx],
                        "response_b": responses_b[idx],
                        "score_a": float(score_array_a[idx]),
                        "score_b": float(score_array_b[idx]),
                    })

            slices.append(SliceComparison(
                name=label.name,
                description=label.description,
                size=len(all_idx),
                failure_rate_a=fr_a,
                failure_rate_b=fr_b,
                failure_rate_diff=fr_a - fr_b,
                concordant_pass=concordant_pass,
                concordant_fail=concordant_fail,
                discordant_a_wins=r.b_count,
                discordant_b_wins=r.c_count,
                advantage_rate=r.advantage_rate,
                p_value=r.p_value,
                adjusted_p_value=r.adjusted_p_value,
                test_used=r.test_used,
                winner=winner,
                sample_indices=all_idx,
                examples=examples,
                representative_prompts=rep_texts[:5],
                cluster_id=r.cluster_id,
            ))

        report = ComparisonReport(
            slices=slices,
            total_prompts=total_prompts,
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            failure_rate_a=failure_rate_a,
            failure_rate_b=failure_rate_b,
            global_p_value=global_result.p_value,
            global_test_used=global_result.test_used,
            global_winner=global_winner,
            global_advantage_rate=global_result.advantage_rate,
            significance_level=self.significance_level,
            failure_threshold=self.failure_threshold,
            scoring_mode=scoring_mode,
            num_clusters_tested=len(corrected),
            num_significant=len(slices),
            clustering_method=self.clustering_method,
            embedding_model=self.embedding_model,
        )

        logger.info(report.summary())
        return report
