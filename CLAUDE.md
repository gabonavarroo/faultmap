# CLAUDE.md — faultmap

## Project Overview

`faultmap` is a pip-installable Python library that automatically discovers WHERE an LLM is failing and WHY, using embedding-space clustering + statistical hypothesis testing. It also audits test suite coverage gaps against production traffic.

**Solo project. 1-week MVP.**

---

## Current Implementation State

### Completed — Phases 1–7

| File | Status |
|------|--------|
| `pyproject.toml` | Done |
| `faultmap/exceptions.py` | Done |
| `faultmap/models.py` | Done (all dataclasses) |
| `faultmap/utils.py` | Done |
| `faultmap/__init__.py` | Done |
| `faultmap/llm.py` | Done |
| `faultmap/embeddings.py` | Done |
| `faultmap/labeling.py` | Done |
| `faultmap/scoring/__init__.py` | Done |
| `faultmap/scoring/base.py` | Done |
| `faultmap/scoring/precomputed.py` | Done |
| `faultmap/scoring/reference.py` | Done |
| `faultmap/scoring/entropy.py` | Done |
| `faultmap/slicing/__init__.py` | Done |
| `faultmap/slicing/clustering.py` | Done |
| `faultmap/slicing/statistics.py` | Done |
| `faultmap/coverage/__init__.py` | Done |
| `faultmap/coverage/detector.py` | Done |
| `faultmap/report.py` | Done (rich + plain text) |
| `faultmap/analyzer.py` | Done (full SliceAnalyzer) |
| `tests/__init__.py` | Done |
| `tests/conftest.py` | Done (`MockEmbedder`, `make_clustered_data`, `make_coverage_data`) |
| `tests/test_utils.py` | Done |
| `tests/test_llm.py` | Done |
| `tests/test_embeddings.py` | Done |
| `tests/test_labeling.py` | Done |
| `tests/test_scoring/` | Done (15 tests) |
| `tests/test_slicing/` | Done (16 tests) |
| `tests/test_coverage/test_detector.py` | Done (6 tests) |
| `tests/test_report.py` | Done (15 tests) |
| `tests/test_analyzer.py` | Done (29 tests) |
| `README.md` | Done (full API docs, scoring modes, examples) |
| `examples/example_mode1_custom_scores.py` | Done |
| `examples/example_mode2_reference_based.py` | Done |
| `examples/example_mode3_reference_free.py` | Done |
| `examples/example_coverage_audit.py` | Done |

**Total: 115 tests, all passing.**
**Coverage: `analyzer.py` 100%, `report.py` 100%.**

### Remaining

```
Phase 7 (PLAN-07): pip install clean install verification (manual step).
```

### conftest.py shared fixtures

Three categories of reusable fixtures now available to all tests:

- **`MockEmbedder`** — deterministic hash-based embedder, `DIM=64`, no model downloads
- **`mock_llm_client`** — `AsyncMock` with canned `"Name: Test Cluster\nDescription: ..."` response
- **`make_clustered_data(n_clusters, n_per_cluster, ...)`** — well-separated cluster embeddings with controllable failure scores; use `failure_clusters=[0]` to inject a known failure pattern
- **`make_coverage_data(n_test, n_prod_covered, n_prod_gap, ...)`** — test + production embeddings in region A (covered) and region B (gap)
- Fixtures `clustered_data`, `small_clustered_data`, `coverage_data` expose the generators as ready-to-use pytest fixtures

### Implementation note — `test_cluster_failure_rate.__test__ = False`

`faultmap/slicing/statistics.py` defines a public function named `test_cluster_failure_rate`. Since pytest collects any module-level name starting with `test_`, we annotate it with `__test__ = False` immediately after definition. This is the standard pytest-supported way to suppress collection of non-test callables.

### Implementation note — coverage gap auto-threshold and float32

`detect_coverage_gaps` uses `mean + 1.5*std` as the auto-threshold. For bimodal distance distributions (clearly separated covered vs. uncovered regions), this can exceed 2.0 (max euclidean distance between unit vectors), detecting nothing. Tests that exercise gap detection use an explicit `distance_threshold=1.0` to avoid this. Similarly, float32 normalization introduces rounding errors that produce tiny non-zero distances; tests for the "all covered" case use float64 arrays.

---

## Implementation Plan Files

All implementation-ready code lives in segmented plan files in the repo root:

| File | Phase | Key contents |
|------|-------|-------------|
| `PLAN.md` | Overview | Architecture, structure, decisions, schedule |
| `PLAN-01-foundation.md` | Day 1 | `pyproject.toml`, `exceptions.py`, `models.py`, `utils.py` |
| `PLAN-02-infrastructure.md` | Day 2 | `llm.py`, `embeddings.py`, `labeling.py` |
| `PLAN-03-scoring.md` | Day 3 | `scoring/` — all three modes |
| `PLAN-04-slicing.md` | Day 4 | `slicing/` — clustering, statistics, BH correction |
| `PLAN-05-integration.md` | Day 5 | `coverage/`, `report.py`, `analyzer.py` |
| `PLAN-06-testing.md` | Day 6 | `conftest.py`, test strategy, e2e scripts |
| `PLAN-07-polish.md` | Days 6-7 | README, examples, packaging, verification |

**Always read the relevant PLAN file before implementing any module.**

---

## Architecture

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

### Scoring Modes

- **Mode 1** (precomputed): user passes `scores=` list — pure passthrough
- **Mode 2** (reference-based): user passes `references=` list — cosine sim scoring
- **Mode 3** (entropy/autonomous): neither passed — semantic entropy + self-consistency via LLM sampling

Mode detection in `analyzer.py`:
- `scores` provided → Mode 1
- `references` provided → Mode 2
- neither → Mode 3
- both → Mode 1 wins, log warning

---

## Key Technical Decisions

### Dependencies

- **litellm** — unified LLM provider (100+ models). No custom provider abstraction.
- **scikit-learn>=1.3** — HDBSCAN is built-in from 1.3+. Do NOT use the standalone `hdbscan` package.
- **No scipy, no statsmodels** — chi-squared via `math.erfc`, Fisher exact via `math.lgamma`, BH in ~20 lines.
- **No pandas** — lists + numpy only. Users can call `.to_dict()` and convert themselves.
- **sentence-transformers** — optional `[local]` extra. Raise `EmbeddingError` with install instructions if missing.
- **rich** — optional `[rich]` extra. Fall back to plain text gracefully.
- **nest-asyncio** — Jupyter compatibility for sync→async bridge.

### Clustering

- **Default**: HDBSCAN (auto-discovers cluster count)
- **Alternative**: agglomerative with silhouette-based k-selection over `[5, 10, 15, 20, 25, 30]`
- **Always L2-normalize** before clustering: `||a-b||² = 2 - 2·cos(a,b)` for unit vectors → euclidean ≈ cosine distance, enables Ward linkage

### Statistical Testing

- **One-sided tests only** — only flag clusters that fail MORE than baseline
- **Decision rule**: expected cell count < 5 → Fisher exact, else chi-squared with Yates correction
- **Chi-squared p-value**: `erfc(sqrt(chi2/2))` (exact for df=1, stdlib only)
- **Fisher exact**: computed via `math.lgamma` to avoid overflow
- **Multiple comparison correction**: Benjamini-Hochberg FDR (not Bonferroni — too conservative)

### Embeddings

- Embed **prompts**, not responses — we want slices of the INPUT space
- Auto-detection in `get_embedder()`: known local prefixes (`all-MiniLM`, `all-mpnet`, `paraphrase-`) → `LocalEmbedder`; otherwise → `APIEmbedder`

### Semantic Entropy (Mode 3)

1. Sample `n_samples` responses per prompt at `temperature=1.0`
2. Embed all samples + original
3. Greedy single-pass clustering within each prompt's samples (threshold=`consistency_threshold`)
4. Shannon entropy of cluster distribution → normalize by `log(n_samples)`
5. Self-consistency = fraction of samples cosine-similar to original (>= `consistency_threshold`)
6. `score = 0.5 * (1 - H_norm) + 0.5 * consistency`

### Coverage Auditing

1. L2-normalize both sets
2. `NearestNeighbors(k=1)` fit on test, query with production
3. Auto-threshold: `mean(distances) + 1.5 * std(distances)` if not specified
4. Cluster uncovered production prompts → gap clusters
5. Name gaps via LLM

---

## Code Style

- Python 3.10+ (`match`, `|` union types, `from __future__ import annotations`)
- `frozen=True` dataclasses for all public models
- Async internally, sync public API via `run_sync()` in `utils.py`
- `logging.getLogger("faultmap")` for all internal logging
- Line length: 99 (ruff)
- Imports: `from __future__ import annotations` at top of every file

---

## Running Tests

```bash
# Install with all deps
pip install -e ".[all]"

# Full test suite
pytest tests/ -v --cov=faultmap --cov-report=term-missing

# Quick smoke test
pytest tests/test_utils.py -v

# Single module
pytest tests/test_slicing/test_statistics.py -v
```

Make sure to activate the venv when running tests with

```bash
source venv/bin/activate
```

**No real API calls in unit tests.** Use `MockEmbedder` and mock `litellm.acompletion` everywhere.

---

## Public API Contract

The public API is frozen and must not change:

```python
from faultmap import SliceAnalyzer

analyzer = SliceAnalyzer(
    model="gpt-4o-mini",
    embedding_model="all-MiniLM-L6-v2",
    significance_level=0.05,
    min_slice_size=10,
    failure_threshold=0.5,
    n_samples=8,
    clustering_method="hdbscan",
    max_concurrent_requests=50,
    temperature=1.0,
)

report = analyzer.analyze(prompts, responses, scores=scores)      # Mode 1
report = analyzer.analyze(prompts, responses, references=refs)   # Mode 2
report = analyzer.analyze(prompts, responses)                     # Mode 3

coverage = analyzer.audit_coverage(test_prompts, production_prompts)
```

`print(report)` must produce readable output. `report.to_dict()` must return JSON-serializable dict.
