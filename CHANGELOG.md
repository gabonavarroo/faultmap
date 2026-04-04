# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/).

---

## [0.4.1] — 2026-04-03
Patch CI Integration tests

## [0.4.0] — 2026-04-03

### Added

**Model comparison**
- `compare_models()` — new public method on `SliceAnalyzer` that compares two models on the same prompt set using paired statistical testing (McNemar's test)
- Supports all three scoring modes: precomputed scores (Mode 1), reference-based (Mode 2), and autonomous entropy scoring (Mode 3)
- Global McNemar test gives a headline winner even when no individual slice reaches significance
- Per-slice McNemar with Benjamini-Hochberg FDR correction identifies which semantic input slices favor each model
- Exact binomial fallback for slices with fewer than 25 discordant pairs (small-sample safe)
- `advantage_rate` effect size: proportion of disagreements where Model A wins (intuitive and actionable)
- `ComparisonReport` frozen dataclass with `__str__()`, `summary()`, `to_dict()` — same pattern as `AnalysisReport`
- `SliceComparison` frozen dataclass with per-slice statistics, discordant pair counts, examples, and winner annotation
- `faultmap/comparison/` subpackage (`statistics.py`) — McNemar's test, exact binomial, BH correction for paired data
- `validate_comparison_inputs()` in `utils.py` — validates lengths, score ranges, and symmetric scores requirement
- Rich table formatter and plain-text formatter for `ComparisonReport`
- `examples/example_model_comparison.py` — standalone example with 3 semantic groups and known per-slice winners

**Exports**
- `ComparisonReport` and `SliceComparison` added to `faultmap.__all__`

---

## [0.3.0] — 2026-04-03

## Fixed

### Embeddings
- Added usage-aware embedding support for asymmetric models, with `query` and `document` routing in the embedding pipeline.
- Added `embedding_usage_kwargs` and `embedding_max_text_chars` to `SliceAnalyzer` for provider-specific embedding configuration and safer handling of long inputs.
- Added default NVIDIA `nv-embedqa` role mapping and automatic API text truncation to reduce embedding failures from strict token limits.

Initial release.

### Added

**Core analysis**
- `SliceAnalyzer` — unified API for all scoring modes and coverage auditing
- `analyze()` — discovers input slices with statistically elevated failure rates
- `audit_coverage()` — finds semantic blind spots in your test suite vs production traffic

**Scoring modes**
- Mode 1 (precomputed): accepts any float scores from DeepEval, Ragas, human labels, or LLM-as-judge
- Mode 2 (reference-based): cosine similarity scoring between responses and ground-truth references
- Mode 3 (entropy/autonomous): semantic entropy + self-consistency via LLM sampling — no labels needed

**Clustering**
- HDBSCAN (default) — auto-discovers cluster count; built into scikit-learn ≥ 1.3
- Agglomerative with silhouette-based k-selection over `[5, 10, 15, 20, 25, 30]` clusters

**Statistical testing**
- One-sided chi-squared (Yates correction) and Fisher exact tests per cluster
- Benjamini-Hochberg FDR correction across all clusters
- Pure stdlib implementation — no scipy or statsmodels dependency

**Embeddings**
- API embeddings via litellm (OpenAI, Voyage, etc.) — default, no extras required
- Local embeddings via sentence-transformers — `pip install faultmap[local]`
- Auto-detection: known local model prefixes route to `LocalEmbedder`, all others to `APIEmbedder`

**Reporting**
- Rich terminal output with tables and color — `pip install faultmap[rich]`
- Plain-text fallback when rich is not installed
- `AnalysisReport.to_dict()` and `CoverageReport.to_dict()` — fully JSON-serializable

**Infrastructure**
- Async-first internally with sync public API via `nest-asyncio`
- Semaphore-limited concurrency for LLM calls (`max_concurrent_requests=50`)
- `logging.getLogger("faultmap")` for structured progress logging

**Developer tooling**
- 130 unit tests, all passing, 95% coverage
- CI on Python 3.10, 3.11, 3.12, 3.13
- OIDC trusted publishing workflow for PyPI
- Tutorial Jupyter notebook with mock and real API paths
- Docker support (`Dockerfile`, `Dockerfile.dev`, `docker-compose.yml`)
