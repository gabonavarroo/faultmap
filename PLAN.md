# faultmap — Master Implementation Plan

> Automatically discover where and why your LLM is failing.

## Context

Existing LLM evaluation tools report aggregate metrics ("85% accuracy") that mask critical failure patterns. `faultmap` is the diagnostic layer that answers two questions:

1. **"Where exactly is my model failing?"** — Embedding-space clustering + statistical testing to find input slices with significantly elevated failure rates.
2. **"Can I trust my test suite?"** — Nearest-neighbor coverage analysis to find semantic blind spots.

## Architecture

```
User API (sync)
    └── SliceAnalyzer
         ├── analyze()         → Score → Embed → Cluster → Test → Correct → Name → Report
         └── audit_coverage()  → Embed → NN Distance → Cluster Gaps → Name → Report

Internal (async via nest_asyncio)
    ├── llm.py          ← litellm wrapper (semaphore rate-limited)
    ├── embeddings.py   ← Local (sentence-transformers) OR API (litellm)
    ├── scoring/        ← 3 modes: precomputed | reference | entropy
    ├── slicing/        ← HDBSCAN/agglomerative + chi2/Fisher + BH correction
    ├── coverage/       ← NN-based gap detection
    ├── labeling.py     ← shared LLM cluster naming
    └── report.py       ← plain text + optional rich formatting
```

## Project Structure

```
faultmap/
├── __init__.py              # Public API exports
├── analyzer.py              # SliceAnalyzer orchestration
├── models.py                # Frozen dataclasses
├── report.py                # Report formatting
├── exceptions.py            # Error hierarchy
├── llm.py                   # Async litellm wrapper
├── embeddings.py            # Embedder ABC + LocalEmbedder + APIEmbedder
├── labeling.py              # LLM cluster naming
├── utils.py                 # Shared utilities
├── scoring/
│   ├── __init__.py
│   ├── base.py              # BaseScorer ABC
│   ├── precomputed.py       # Mode 1
│   ├── reference.py         # Mode 2
│   └── entropy.py           # Mode 3
├── slicing/
│   ├── __init__.py
│   ├── clustering.py        # HDBSCAN + agglomerative
│   └── statistics.py        # Statistical tests + BH correction
└── coverage/
    ├── __init__.py
    └── detector.py           # NN-based gap detection

tests/
├── conftest.py
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
└── test_analyzer.py
```

## Segmented Plan Files

| File | Phase | Contents |
|------|-------|----------|
| [PLAN-01-foundation.md](PLAN-01-foundation.md) | Day 1 | `pyproject.toml`, `exceptions.py`, `models.py`, `utils.py` |
| [PLAN-02-infrastructure.md](PLAN-02-infrastructure.md) | Day 2 | `llm.py`, `embeddings.py`, `labeling.py` |
| [PLAN-03-scoring.md](PLAN-03-scoring.md) | Day 3 | `scoring/` — all three modes with full algorithms |
| [PLAN-04-slicing.md](PLAN-04-slicing.md) | Day 4 | `slicing/` — clustering, statistics, BH correction |
| [PLAN-05-integration.md](PLAN-05-integration.md) | Day 5 | `coverage/`, `report.py`, `analyzer.py` orchestration |
| [PLAN-06-testing.md](PLAN-06-testing.md) | Testing | Full test strategy, fixtures, synthetic data |
| [PLAN-07-polish.md](PLAN-07-polish.md) | Days 6-7 | README, examples, packaging, verification |

## Key Design Decisions

1. **Embed prompts, not responses** — slices of input space where the model struggles
2. **One-sided tests** — only flag clusters that fail MORE than baseline
3. **BH over Bonferroni** — FDR control, not FWER; standard for multi-hypothesis testing
4. **No scipy/statsmodels** — chi2 via `math.erfc`, Fisher via `math.lgamma`, BH in ~20 lines
5. **L2 normalize → euclidean = cosine** — enables Ward linkage on cosine distances
6. **HDBSCAN from sklearn 1.3+** — no standalone C-extension package needed
7. **litellm** — single dep for 100+ LLM providers, no custom abstraction
8. **Sync API, async internals** — `run_sync()` + `nest_asyncio` for Jupyter compat
9. **sentence-transformers optional** — `pip install faultmap[local]` for local embeddings

## Dependencies

**Required**: `numpy>=1.24`, `scikit-learn>=1.3`, `litellm>=1.30`, `tqdm>=4.60`, `nest-asyncio>=1.5`
**Optional**: `sentence-transformers>=2.2` (`[local]`), `rich>=13.0` (`[rich]`)
**Dev**: `pytest>=7.0`, `pytest-asyncio>=0.21`, `pytest-cov>=4.0`, `ruff>=0.4`

## Schedule

| Day | Phase | Milestone |
|-----|-------|-----------|
| 1 | Foundation | Package installs, models defined, utils work |
| 2 | Infrastructure | LLM calls + embeddings (both backends) work |
| 3 | Scoring | All 3 scoring modes work independently |
| 4 | Slicing | Cluster + test + correct + name pipeline works |
| 5 | Integration | Full `analyze()` and `audit_coverage()` end-to-end |
| 6 | Polish | Real e2e test, README, packaging verified |
| 7 | Buffer | Overflow, docstrings, examples, final review |
