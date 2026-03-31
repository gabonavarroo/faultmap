# faultmap — Implementation Plan

## Context

**Problem**: Existing LLM evaluation tools report aggregate metrics ("85% accuracy") that mask critical failure patterns. No tool automatically discovers *where* failures concentrate or audits test suite coverage against production traffic.

**Solution**: `faultmap` — a pip-installable Python library that uses embedding-space clustering + statistical testing to find failure slices, and nearest-neighbor analysis to find coverage gaps. Operates in 3 scoring modes (bring-your-own, reference-based, or fully autonomous via semantic entropy).

**Outcome**: ML engineers can identify statistically significant failure patterns before deployment, debug production issues faster, and build better test suites.

---

## Architecture Overview

```
User API (sync)
    └── SliceAnalyzer
         ├── analyze() ──→ Score → Embed → Cluster → Test → Correct → Name → Report
         └── audit_coverage() ──→ Embed → NN Distance → Cluster Gaps → Name → Report

Internal (async)
    ├── llm.py          ← litellm wrapper (rate-limited async)
    ├── embeddings.py   ← Local (sentence-transformers) or API (litellm)
    ├── scoring/        ← 3 modes: precomputed, reference, entropy
    ├── slicing/        ← clustering + statistical tests + BH correction
    ├── coverage/       ← NN-based gap detection
    ├── labeling.py     ← shared LLM cluster naming
    └── report.py       ← plain text + optional rich formatting
```

---

## Project Structure

```
faultmap/
├── __init__.py              # Public API: SliceAnalyzer, model classes, exceptions
├── analyzer.py              # SliceAnalyzer orchestration (sync → async bridge)
├── models.py                # Frozen dataclasses: reports, slices, gaps
├── report.py                # Report formatting (plain text + optional rich)
├── exceptions.py            # FaultmapError hierarchy
├── llm.py                   # Async litellm wrapper with semaphore rate limiting
├── embeddings.py            # Embedder ABC + LocalEmbedder + APIEmbedder + factory
├── labeling.py              # LLM-assisted cluster naming (shared)
├── scoring/
│   ├── __init__.py
│   ├── base.py              # BaseScorer ABC
│   ├── precomputed.py       # Mode 1: passthrough
│   ├── reference.py         # Mode 2: cosine similarity
│   └── entropy.py           # Mode 3: semantic entropy + self-consistency
├── slicing/
│   ├── __init__.py
│   ├── clustering.py        # HDBSCAN (default) + agglomerative
│   └── statistics.py        # Chi-squared, Fisher exact, BH correction
├── coverage/
│   ├── __init__.py
│   └── detector.py          # NN-based gap detection
└── utils.py                 # cosine similarity, run_sync, validation, batching

tests/
├── conftest.py              # Shared fixtures: mock embedder, mock LLM, synthetic data
├── test_utils.py
├── test_embeddings.py
├── test_llm.py
├── test_labeling.py
├── test_scoring/
│   ├── test_precomputed.py
│   ├── test_reference.py
│   └── test_entropy.py
├── test_slicing/
│   ├── test_clustering.py
│   └── test_statistics.py
├── test_coverage/
│   └── test_detector.py
├── test_report.py
└── test_analyzer.py         # Integration tests

examples/
├── example_mode1_custom_scores.py
├── example_mode2_reference_based.py
├── example_mode3_reference_free.py
└── example_coverage_audit.py
```

---

## Dependencies

### pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "faultmap"
version = "0.1.0"
description = "Automatically discover where and why your LLM is failing"
readme = "README.md"
license = "Apache-2.0"
requires-python = ">=3.10"
authors = [{ name = "Gabriel Navarro" }]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "numpy>=1.24",
    "scikit-learn>=1.3",
    "litellm>=1.30",
    "tqdm>=4.60",
    "nest-asyncio>=1.5",
]

[project.optional-dependencies]
local = ["sentence-transformers>=2.2"]
rich = ["rich>=13.0"]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "pytest-cov>=4.0",
    "ruff>=0.4",
]
all = ["faultmap[local,rich,dev]"]

[tool.ruff]
target-version = "py310"
line-length = 99

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

**Dependency rationale**:
- `numpy` — array ops for embeddings, cosine similarity, entropy math
- `scikit-learn>=1.3` — HDBSCAN, AgglomerativeClustering, NearestNeighbors (HDBSCAN added in 1.3)
- `litellm` — unified LLM provider (100+ models), also handles API-based embeddings
- `tqdm` — progress bars for long-running async operations
- `nest-asyncio` — Jupyter compatibility for sync→async bridge
- **No scipy** — chi-squared and Fisher implemented via `math.erfc`/`math.lgamma` (stdlib)
- **No statsmodels** — BH correction is ~20 lines
- **No pandas** — lists + numpy suffice; users can convert via `.to_dict()`
- `sentence-transformers` (optional `[local]`) — local embedding models
- `rich` (optional `[rich]`) — pretty table output

---

## Module Specifications

### 1. `exceptions.py`

```python
class FaultmapError(Exception): ...
class EmbeddingError(FaultmapError): ...
class ScoringError(FaultmapError): ...
class LLMError(FaultmapError): ...
class ClusteringError(FaultmapError): ...
class ConfigurationError(FaultmapError): ...
```

### 2. `models.py` — Complete Specification

All dataclasses are `frozen=True` for immutability.

```python
@dataclass(frozen=True)
class ScoringResult:
    scores: list[float]           # Per-prompt score in [0, 1]. Higher = better.
    mode: str                     # "precomputed" | "reference" | "entropy"
    metadata: dict                # Mode-specific (entropy values, etc.)

@dataclass(frozen=True)
class FailureSlice:
    name: str                     # LLM-generated name ("Legal compliance questions")
    description: str              # LLM-generated 1-sentence explanation
    size: int                     # Number of prompts in this slice
    failure_rate: float           # Failure rate within this slice
    baseline_rate: float          # Overall failure rate
    effect_size: float            # failure_rate / baseline_rate (risk ratio)
    p_value: float                # Raw p-value
    adjusted_p_value: float       # BH-corrected p-value
    test_used: str                # "chi2" | "fisher"
    sample_indices: list[int]     # ALL indices in the cluster (into original prompts)
    examples: list[dict]          # Top-5 dicts: {"prompt": str, "response": str, "score": float}
    representative_prompts: list[str]  # Top-5 prompts closest to centroid
    cluster_id: int               # Internal cluster label

@dataclass(frozen=True)
class AnalysisReport:
    slices: list[FailureSlice]    # Sorted by adjusted_p_value ascending
    total_prompts: int
    total_failures: int
    baseline_failure_rate: float
    significance_level: float
    failure_threshold: float
    scoring_mode: str
    num_clusters_tested: int
    num_significant: int
    clustering_method: str
    embedding_model: str
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str: ...       # 1-paragraph plain text
    def to_dict(self) -> dict: ...      # JSON-serializable
    def __str__(self) -> str: ...       # Calls format_analysis_report()

@dataclass(frozen=True)
class CoverageGap:
    name: str                     # "Technical setup questions"
    description: str
    size: int                     # Production prompts in this gap
    mean_distance: float          # Avg NN distance to nearest test case
    representative_prompts: list[str]
    prompt_indices: list[int]     # Indices into production_prompts
    cluster_id: int

@dataclass(frozen=True)
class CoverageReport:
    gaps: list[CoverageGap]       # Sorted by mean_distance descending
    num_test_prompts: int
    num_production_prompts: int
    num_gaps: int
    overall_coverage_score: float # 1 - (fraction of prod prompts in gaps)
    distance_threshold: float
    embedding_model: str
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str: ...
    def to_dict(self) -> dict: ...
    def __str__(self) -> str: ...       # Calls format_coverage_report()
```

**Key**: `AnalysisReport.__str__` and `CoverageReport.__str__` call the report formatter so that `print(report)` produces a pretty-printed output.

### 3. `utils.py`

```python
def cosine_similarity_matrix(a: ndarray, b: ndarray) -> ndarray:
    """(n, d) x (m, d) → (n, m) cosine similarity matrix."""

def cosine_similarity_pairs(a: ndarray, b: ndarray) -> ndarray:
    """(n, d) x (n, d) → (n,) element-wise cosine similarity."""

def run_sync(coro: Awaitable[T]) -> T:
    """Run async coroutine synchronously. Handles Jupyter via nest_asyncio."""
    # Try get_running_loop → nest_asyncio.apply() + run_until_complete
    # Fallback → asyncio.run()

def validate_inputs(prompts, responses, scores, references) -> None:
    """Validate lengths, types, score ranges. Raises ConfigurationError."""

def batch_items(items: list[T], batch_size: int) -> list[list[T]]:
    """Split list into batches."""
```

### 4. `llm.py` — Async LLM Client

```python
class AsyncLLMClient:
    def __init__(self, model: str, max_concurrent_requests: int = 50,
                 max_retries: int = 3, timeout: float = 60.0):
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def complete(self, messages: list[dict], temperature=0.0,
                       max_tokens=512) -> str:
        """Single completion with semaphore + exponential backoff retry."""
        # Acquire semaphore → litellm.acompletion() → retry on failure
        # Raises LLMError after max_retries

    async def complete_batch(self, messages_list: list[list[dict]],
                             temperature=0.0, desc="LLM calls",
                             show_progress=True) -> list[str]:
        """Batch completions with tqdm progress bar via tqdm_asyncio.gather()."""
```

### 5. `embeddings.py` — Dual Backend

```python
class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray: ...  # (n, d)
    @property
    @abstractmethod
    def dimension(self) -> int: ...

class LocalEmbedder(Embedder):
    """sentence-transformers backend. Lazy model loading."""
    LOCAL_MODEL_PREFIXES = ("all-MiniLM", "all-mpnet", "paraphrase-", ...)
    # embed(): loads model on first call, calls model.encode()
    # Raises EmbeddingError with install instructions if sentence-transformers missing

class APIEmbedder(Embedder):
    """litellm.embedding() backend. Handles batching."""
    # embed(): batches texts, calls litellm.embedding(), extracts vectors

def get_embedder(model_name: str) -> Embedder:
    """Factory: auto-detect local vs API based on model name prefixes."""
    # Known local prefixes → LocalEmbedder
    # Known HuggingFace orgs → LocalEmbedder
    # Otherwise → APIEmbedder
```

### 6. `labeling.py` — Shared Cluster Naming

```python
@dataclass(frozen=True)
class ClusterLabel:
    name: str          # 2-5 word name
    description: str   # 1-sentence description

async def label_cluster(client: AsyncLLMClient, representative_texts: list[str],
                        context: str = "failure slice") -> ClusterLabel:
    """Ask LLM to name a cluster. Parse "Name: ...\nDescription: ..." format."""

async def label_clusters(client: AsyncLLMClient, clusters_texts: list[list[str]],
                         context: str) -> list[ClusterLabel]:
    """Label multiple clusters concurrently via asyncio.gather."""
```

### 7. `scoring/` — Three Modes

**`base.py`**: Abstract `BaseScorer` with `async def score(prompts, responses, **kwargs) -> ScoringResult`

**`precomputed.py`** (Mode 1): Pure passthrough. Returns user scores as-is.

**`reference.py`** (Mode 2):
1. Embed responses and references
2. `score = (cosine_similarity_pairs(resp_emb, ref_emb) + 1) / 2` — maps [-1,1] to [0,1]

**`entropy.py`** (Mode 3 — most complex):

```
Algorithm:
1. SAMPLE: For each of N prompts, sample n_samples responses at high temperature
   → n_samples * N total LLM calls via complete_batch() (rate-limited, parallel)

2. EMBED: Batch-embed all sampled responses + original responses in one call
   → Reshape to (N, n_samples, d) for samples + (N, d) for originals

3. SEMANTIC ENTROPY per prompt:
   a. Compute pairwise cosine sim matrix among n_samples embeddings
   b. Greedy clustering: pick unassigned sample as center, assign all
      unassigned samples with sim >= consistency_threshold to same cluster
   c. Cluster probabilities: p_k = |cluster_k| / n_samples
   d. Entropy H = -Σ p_k * log(p_k)
   e. Normalize: H_norm = H / log(n_samples), clamp to [0, 1]

4. SELF-CONSISTENCY per prompt:
   a. Cosine sim between original response embedding and each sample
   b. Fraction of samples with sim >= consistency_threshold

5. COMBINE: score = 0.5 * (1 - H_norm) + 0.5 * self_consistency
   → High score = low entropy + high agreement = reliable
   → Low score = high entropy + low agreement = uncertain/hallucinating
```

### 8. `slicing/clustering.py`

```python
def cluster_embeddings(embeddings: ndarray, method="hdbscan",
                       min_cluster_size=10) -> ndarray:
    """
    Returns labels array. -1 = noise/removed.

    HDBSCAN (default):
    1. L2-normalize embeddings (euclidean on unit vectors ≈ cosine distance)
    2. sklearn.cluster.HDBSCAN(min_cluster_size, metric="euclidean", method="eom")
    3. Error if all points are noise

    Agglomerative:
    1. L2-normalize
    2. Auto-select n_clusters via silhouette score over [5, 10, 15, 20, 25, 30]
    3. AgglomerativeClustering(n_clusters, linkage="ward")
    4. Post-filter: set labels to -1 for clusters smaller than min_cluster_size
    """

def get_representative_prompts(embeddings, labels, cluster_id, prompts,
                               top_k=5) -> tuple[list[str], list[int]]:
    """Top-k prompts closest to cluster centroid by cosine similarity."""
```

### 9. `slicing/statistics.py`

```python
@dataclass
class ClusterTestResult:
    cluster_id: int
    size: int
    failure_count: int
    failure_rate: float
    p_value: float
    test_used: str               # "chi2" | "fisher" | "none"
    adjusted_p_value: float = 1.0  # Set after BH correction

def test_cluster_failure_rate(cluster_failures, cluster_size,
                              total_failures, total_size,
                              cluster_id) -> ClusterTestResult:
    """
    2x2 contingency table test. ONE-SIDED (only care if failure rate is HIGHER).

    Decision: if any expected cell count < 5 → Fisher exact test, else chi-squared.

    Chi-squared with Yates correction:
      chi2 = N * (|ad - bc| - N/2)^2 / ((a+b)(c+d)(a+c)(b+d))
      p_two_sided = erfc(sqrt(chi2/2))    # exact for df=1, uses math.erfc
      p_one_sided = p_two_sided / 2

    Fisher exact (one-sided greater):
      P(X >= a) where X ~ Hypergeometric(N, K, n)
      Computed via log-gamma to avoid overflow

    Skip test if cluster failure rate <= baseline (p = 1.0).
    """

def benjamini_hochberg(results: list[ClusterTestResult],
                       alpha: float = 0.05) -> list[ClusterTestResult]:
    """
    BH FDR correction:
    1. Sort by p-value ascending
    2. adjusted[i] = p[i] * m / rank[i]
    3. Enforce monotonicity (backward cumulative min)
    4. Clip to [0, 1]
    Returns sorted by adjusted_p_value ascending.
    """
```

### 10. `coverage/detector.py`

```python
def detect_coverage_gaps(test_embeddings, prod_embeddings, prod_prompts,
                         distance_threshold=None, min_gap_size=5,
                         clustering_method="hdbscan"):
    """
    1. L2-normalize both sets
    2. Fit NearestNeighbors(k=1) on test embeddings
    3. Query with prod embeddings → distances
    4. Auto-threshold if None: mean + 1.5 * std
    5. Uncovered = distance > threshold
    6. Cluster uncovered embeddings → gap clusters
    Returns: (gap_labels, nn_distances, threshold)
    """
```

### 11. `report.py`

```python
def format_analysis_report(report: AnalysisReport) -> str:
    """Try rich tables, fall back to plain text."""

def format_coverage_report(report: CoverageReport) -> str:
    """Try rich tables, fall back to plain text."""
```

Plain text format uses `═` / `─` box drawing for visual structure. Rich format uses `rich.table.Table`.

### 12. `analyzer.py` — The Orchestrator

```python
class SliceAnalyzer:
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
    ):
        # Validate config params
        # self._embedder = get_embedder(embedding_model)
        # self._llm_client = AsyncLLMClient(model, max_concurrent_requests)

    def analyze(self, prompts, responses, scores=None, references=None) -> AnalysisReport:
        """Sync entry point. Delegates to _analyze_async via run_sync()."""
        # Mode detection:
        #   scores provided → Mode 1
        #   references provided → Mode 2
        #   neither → Mode 3
        #   both → scores wins with warning

    async def _analyze_async(self, prompts, responses, scores, references):
        """
        Pipeline:
        1. VALIDATE inputs
        2. DETECT mode, create scorer
        3. SCORE → ScoringResult
        4. BINARIZE → failures = score < threshold
        5. Early return if 0 failures
        6. EMBED prompts (NOT responses — cluster input space)
        7. CLUSTER → labels array
        8. TEST each cluster → ClusterTestResult list
        9. BH CORRECT → adjusted p-values
        10. FILTER → keep adjusted_p < alpha
        11. NAME significant clusters via LLM
        12. ASSEMBLE FailureSlice objects with:
            - ALL sample_indices (every prompt in the cluster)
            - examples: top-5 dicts {prompt, response, score}
            - effect_size: failure_rate / baseline_rate
        13. Return AnalysisReport
        """

    def audit_coverage(self, test_prompts, production_prompts,
                       distance_threshold=None, min_gap_size=5) -> CoverageReport:
        """Sync entry point for coverage auditing."""

    async def _audit_coverage_async(self, ...):
        """
        Pipeline:
        1. VALIDATE inputs
        2. EMBED test + production prompts
        3. DETECT GAPS → gap_labels, distances, threshold
        4. GET representatives per gap cluster
        5. NAME gaps via LLM
        6. ASSEMBLE CoverageGap objects, CoverageReport
        """
```

---

## Key Design Decisions

1. **Embed prompts, not responses** — We want slices of *input space* where the model struggles. Responses vary too much to cluster meaningfully.

2. **One-sided tests only** — We only flag clusters that fail *more* than baseline. Clusters that fail less are not interesting for fault-finding.

3. **BH over Bonferroni** — With potentially dozens of clusters, Bonferroni is too conservative. BH controls false discovery rate, standard in genomics and ML.

4. **No scipy/statsmodels** — Chi-squared p-value for df=1 is `erfc(sqrt(x/2))` (stdlib `math.erfc`). Fisher exact uses `math.lgamma`. BH is ~20 lines. This eliminates two heavy dependencies.

5. **L2 normalize before clustering** — `||a-b||² = 2 - 2·cos(a,b)` for unit vectors. Euclidean on normalized vectors = cosine distance. Enables Ward linkage.

6. **HDBSCAN from sklearn 1.3+** — Avoids standalone `hdbscan` package with C extension issues.

7. **Greedy clustering for semantic entropy** — Full clustering algorithm is overkill for N=8 samples. Greedy single-pass is O(n²) but n is tiny.

8. **litellm for all LLM calls** — Single dependency for 100+ models. No custom provider abstraction needed.

9. **`(cosine_sim + 1) / 2` mapping** in Mode 2 — Ensures scores are always in [0, 1].

10. **Sync API, async internals** — `analyze()` is sync. Internally uses `asyncio.run()` + `nest_asyncio` for Jupyter.

---

## Test Strategy

### Mocking approach
- **Embeddings**: `MockEmbedder` returns deterministic vectors based on text hashing (64-dim). No model download.
- **LLM calls**: Mock `litellm.acompletion` to return canned responses. Use `pytest-asyncio` for async tests.
- **No real API calls** in unit tests. Integration tests (manual, Day 6) use real APIs.

### Per-module test plan

| Module | Key tests |
|--------|-----------|
| `utils.py` | cosine similarity with orthogonal/parallel vectors; validation error cases |
| `embeddings.py` | LocalEmbedder raises EmbeddingError when not installed; APIEmbedder batching (300 texts → 3 API calls); `get_embedder` routing |
| `llm.py` | Retry logic (fail 2x, succeed 3rd); semaphore concurrency; batch ordering |
| `labeling.py` | Parse well-formed/malformed "Name: X\nDescription: Y" responses |
| `scoring/precomputed` | Passthrough verification |
| `scoring/reference` | Known vectors → known cosine sim → known score |
| `scoring/entropy` | **Case 1**: all identical samples → entropy=0, consistency=1, score≈1. **Case 2**: all orthogonal → max entropy, consistency≈0, score≈0. **Case 3**: 2 clusters of 4 → moderate score |
| `slicing/clustering` | 3 well-separated Gaussian clusters; HDBSCAN finds ~3; agglomerative with silhouette; edge: too few points |
| `slicing/statistics` | Known contingency tables with analytically verified p-values; Fisher vs chi2 decision; BH: 5 p-values → known adjusted values; edge: all pass, all fail |
| `coverage/detector` | Test embeddings in region A, prod in A+B. Gap detected in B. Auto-threshold. Edge: full coverage |
| `report.py` | Key strings present in formatted output; empty slices case |
| `analyzer.py` | **Integration**: mock embedder + LLM, full pipeline Mode 1 with synthetic data having known failure cluster → verify it's discovered. Mode detection (scores > references). Coverage end-to-end |

### Synthetic data generator (conftest.py)

```python
def make_clustered_data(n_clusters=3, n_per_cluster=30, dim=64, failure_cluster=0):
    """
    Generate prompts, responses, scores with a known failure pattern.
    Cluster 0 has low scores (failures), clusters 1-2 have high scores.
    Returns (prompts, responses, scores, embeddings).
    """
```

---

## Implementation Schedule

### Day 1 — Foundation
**Files**: `pyproject.toml`, `faultmap/__init__.py`, `exceptions.py`, `models.py`, `utils.py`
**Tests**: `test_utils.py`
**Milestone**: Package installs via `pip install -e .`, models defined, utils work.

### Day 2 — LLM + Embeddings
**Files**: `llm.py`, `embeddings.py`
**Tests**: `test_llm.py`, `test_embeddings.py`
**Milestone**: Can make async LLM calls and compute embeddings (both backends).

### Day 3 — Scoring Pipeline
**Files**: `scoring/base.py`, `scoring/__init__.py`, `scoring/precomputed.py`, `scoring/reference.py`, `scoring/entropy.py`
**Tests**: `test_scoring/` (all three)
**Milestone**: All three scoring modes work independently with mocked deps.

### Day 4 — Slicing Pipeline
**Files**: `slicing/clustering.py`, `slicing/statistics.py`, `slicing/__init__.py`, `labeling.py`
**Tests**: `test_slicing/`, `test_labeling.py`
**Milestone**: Can cluster, test, correct, name. Statistical tests verified.

### Day 5 — Coverage + Reports + Orchestrator
**Files**: `coverage/detector.py`, `coverage/__init__.py`, `report.py`, `analyzer.py`
**Tests**: `test_coverage/`, `test_report.py`, `test_analyzer.py`
**Milestone**: Full `analyze()` and `audit_coverage()` work end-to-end.

### Day 6 — Integration + Polish
- Manual e2e test with real gpt-4o-mini + real embeddings
- Fix issues found
- README with usage examples
- Verify `pip install .` and `pip install .[local]`

### Day 7 — Buffer + Ship
- Handle overflow from prior days
- Docstrings on all public API
- Example scripts in `examples/`
- Final review

---

## README Structure

```
# faultmap

> Automatically discover where and why your LLM is failing.

[badges: PyPI, Python, License, Tests]

## The Problem
Your eval says 85% accuracy. Users are complaining.
Aggregate metrics hide failure patterns.

## What faultmap Does
[30-second explanation + diagram of the 3 modes]

## Installation
pip install faultmap          # API embeddings
pip install faultmap[local]   # + sentence-transformers for local embeddings
pip install faultmap[all]     # everything

## Quick Start
[3 code blocks — one per mode, each ~10 lines]

## Coverage Auditing
[Single code block]

## How It Works
[Brief algorithm overview: embed → cluster → test → correct → name]

## API Reference
[SliceAnalyzer constructor params, analyze(), audit_coverage()]

## License
Apache 2.0
```

---

## Verification Plan

After implementation, verify with:

1. **Unit tests pass**: `pytest tests/ -v`
2. **Package installs**: `pip install -e .` and `pip install -e ".[local,rich]"`
3. **Mode 1 e2e**: Create 100 prompts with known failure cluster, provide scores, verify slice discovered
4. **Mode 2 e2e**: Same dataset with reference answers, verify cosine-based scoring works
5. **Mode 3 e2e**: Small dataset (20 prompts), real LLM sampling, verify entropy scores are reasonable
6. **Coverage e2e**: Test set missing a category, production set has it, verify gap detected
7. **`print(report)`** produces readable output
8. **Edge cases**: empty slices, all pass, single cluster, small datasets

---

## Post-MVP (Architected For, Not Implemented)

- `comparison/model_compare.py` — side-by-side slice analysis
- Temporal tracking — compare slices across runs
- Export slices as test datasets
- Async public API (`async def analyze_async()`)
- Caching layer (embeddings, LLM responses)
- Direct DeepEval/Ragas adapters
- Web dashboard
