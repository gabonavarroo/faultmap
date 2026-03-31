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
- [~] `faultmap/analyzer.py` — empty class stub
- [~] `faultmap/report.py` — stub (calls summary())
- [x] `tests/test_utils.py`
- [~] `tests/conftest.py` — minimal stub, needs full fixtures from PLAN-06

## Phase 2 — Infrastructure (Day 2)

- [x] `faultmap/llm.py`
- [x] `faultmap/embeddings.py`
- [x] `faultmap/labeling.py`
- [x] `tests/test_llm.py`
- [x] `tests/test_embeddings.py`
- [x] `tests/test_labeling.py`

## Phase 3 — Scoring (Day 3)

- [ ] `faultmap/scoring/__init__.py`
- [ ] `faultmap/scoring/base.py`
- [ ] `faultmap/scoring/precomputed.py`
- [ ] `faultmap/scoring/reference.py`
- [ ] `faultmap/scoring/entropy.py`
- [ ] `tests/test_scoring/__init__.py`
- [ ] `tests/test_scoring/test_precomputed.py`
- [ ] `tests/test_scoring/test_reference.py`
- [ ] `tests/test_scoring/test_entropy.py`

## Phase 4 — Slicing (Day 4)

- [ ] `faultmap/slicing/__init__.py`
- [ ] `faultmap/slicing/clustering.py`
- [ ] `faultmap/slicing/statistics.py`
- [ ] `tests/test_slicing/__init__.py`
- [ ] `tests/test_slicing/test_clustering.py`
- [ ] `tests/test_slicing/test_statistics.py`

## Phase 5 — Integration (Day 5)

- [ ] `faultmap/coverage/__init__.py`
- [ ] `faultmap/coverage/detector.py`
- [~] `faultmap/report.py` — needs full implementation (rich + plain)
- [~] `faultmap/analyzer.py` — needs full pipeline
- [ ] `tests/test_coverage/__init__.py`
- [ ] `tests/test_coverage/test_detector.py`
- [ ] `tests/test_report.py`
- [ ] `tests/test_analyzer.py`

## Phase 6 — Testing (Day 6)

- [~] `tests/conftest.py` — needs MockEmbedder, make_clustered_data, make_coverage_data
- [ ] Manual e2e test: Mode 1 with real gpt-4o-mini
- [ ] Manual e2e test: Coverage auditing with real gpt-4o-mini

## Phase 7 — Polish (Days 6-7)

- [ ] `README.md` — rewrite from placeholder
- [ ] `examples/example_mode1_custom_scores.py`
- [ ] `examples/example_mode2_reference_based.py`
- [ ] `examples/example_mode3_reference_free.py`
- [ ] `examples/example_coverage_audit.py`
- [ ] Docstrings on all public API
- [ ] `pip install faultmap` clean install verified
- [ ] `pytest tests/ -v` passes 100%

---

## Test Command Reference

```bash
pip install -e ".[all]"
pytest tests/ -v --cov=faultmap --cov-report=term-missing
```
