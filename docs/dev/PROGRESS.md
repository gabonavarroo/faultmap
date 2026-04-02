# Implementation Progress

Track which files are done, stubbed, or not started. Update this as you go.

## Legend
- `[x]` Done — fully implemented and tested
- `[~]` Partial — stub or incomplete
- `[ ]` Not started

---

## Phase 1 — Foundation (Day 1)

- [x] `pyproject.toml`
- [x] `faultmap/exceptions.py`
- [x] `faultmap/models.py`
- [x] `faultmap/utils.py`
- [x] `faultmap/__init__.py`
- [x] `tests/test_utils.py`

## Phase 2 — Infrastructure (Day 2)

- [x] `faultmap/llm.py`
- [x] `faultmap/embeddings.py`
- [x] `faultmap/labeling.py`
- [x] `tests/test_llm.py`
- [x] `tests/test_embeddings.py`
- [x] `tests/test_labeling.py`

## Phase 3 — Scoring (Day 3)

- [x] `faultmap/scoring/__init__.py`
- [x] `faultmap/scoring/base.py`
- [x] `faultmap/scoring/precomputed.py`
- [x] `faultmap/scoring/reference.py`
- [x] `faultmap/scoring/entropy.py`
- [x] `tests/test_scoring/__init__.py`
- [x] `tests/test_scoring/test_precomputed.py`
- [x] `tests/test_scoring/test_reference.py`
- [x] `tests/test_scoring/test_entropy.py`

## Phase 4 — Slicing (Day 4)

- [x] `faultmap/slicing/__init__.py`
- [x] `faultmap/slicing/clustering.py`
- [x] `faultmap/slicing/statistics.py`
- [x] `tests/test_slicing/__init__.py`
- [x] `tests/test_slicing/test_clustering.py`
- [x] `tests/test_slicing/test_statistics.py`

## Phase 5 — Integration (Day 5)

- [x] `faultmap/coverage/__init__.py`
- [x] `faultmap/coverage/detector.py`
- [x] `faultmap/report.py` — rich + plain text dual formatting
- [x] `faultmap/analyzer.py` — full `SliceAnalyzer` (analyze + audit_coverage)
- [x] `tests/test_coverage/__init__.py`
- [x] `tests/test_coverage/test_detector.py`
- [x] `tests/test_report.py`
- [x] `tests/test_analyzer.py`

## Phase 6 — Testing (Day 6)

- [x] `tests/conftest.py` — `MockEmbedder`, `make_clustered_data`, `make_coverage_data`, fixtures
- [x] `tests/test_analyzer.py` — expanded: Mode 2, Mode 3, full pipeline, `_audit_coverage_async`, HDBSCAN noise (29 tests, 100% coverage)
- [x] `tests/test_report.py` — expanded: plain text, rich with-gaps, ImportError fallback (15 tests, 100% coverage)
- [ ] Manual e2e test: Mode 1 with real gpt-4o-mini
- [ ] Manual e2e test: Coverage auditing with real gpt-4o-mini

**Current test count: 115 passing. analyzer.py: 100%, report.py: 100%.**

### Note — `tests/__init__.py` removed

Adding `tests/__init__.py` causes pytest to import `conftest.py` both as a conftest
and as a package module, triggering numpy's "cannot load module more than once"
error. The file was removed; subdirectory `__init__.py` files (test_scoring, etc.)
are fine and remain.

## Phase 7 — Polish (Days 6-7)

- [x] `README.md` — rewritten with full docs, API reference, scoring modes, examples
- [x] `examples/example_mode1_custom_scores.py`
- [x] `examples/example_mode2_reference_based.py`
- [x] `examples/example_mode3_reference_free.py`
- [x] `examples/example_coverage_audit.py`
- [x] Docstrings on all public API (`SliceAnalyzer`, `AnalysisReport`, `CoverageReport`, `FailureSlice`, `CoverageGap`, `ScoringResult`)
- [ ] `pip install faultmap` clean install verified
- [x] `pytest tests/ -v` passes 100%

---

## Test Command Reference

```bash
pip install -e ".[all]"
pytest tests/ -v --cov=faultmap --cov-report=term-missing
```
