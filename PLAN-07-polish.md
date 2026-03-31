# Phase 7 — Polish & Ship (Days 6-7)

**Goal**: README, examples, packaging verification, final review.

**Files to create/update**:
- `README.md` (rewrite from placeholder)
- `examples/example_mode1_custom_scores.py`
- `examples/example_mode2_reference_based.py`
- `examples/example_mode3_reference_free.py`
- `examples/example_coverage_audit.py`
- Docstrings on all public API

**Milestone**: Library is pip-installable, documented, and ready for early users.

---

## 1. README.md

```markdown
# faultmap

> Automatically discover where and why your LLM is failing.

[![PyPI](https://img.shields.io/pypi/v/faultmap)](https://pypi.org/project/faultmap/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

---

## The Problem

Your eval says **85% accuracy**. Users are complaining. Where are the failures?

Aggregate metrics hide critical patterns. A model can score well overall while
catastrophically failing on specific input types — legal questions, billing
disputes, technical setup — that matter most to your users.

**faultmap** is the diagnostic layer that answers two questions:

1. **"Where exactly is my model failing?"** — Automated discovery of input
   slices where failure rate is statistically significantly higher than baseline.
2. **"Can I trust my test suite?"** — Embedding-space coverage analysis that
   finds semantic blind spots your test suite has never touched.

## How It Works

```
Prompts → Embed → Cluster → Statistical Test → BH Correction → Name → Report
```

faultmap embeds your prompts into a semantic space, clusters them to find
coherent input groups, runs hypothesis tests to identify clusters with elevated
failure rates, applies Benjamini-Hochberg correction to control false discovery
rate, and uses an LLM to generate human-readable names for each failure slice.

## Installation

```bash
pip install faultmap                # Core (uses API embeddings)
pip install faultmap[local]         # + sentence-transformers for local embeddings
pip install faultmap[rich]          # + rich for pretty terminal output
pip install faultmap[all]           # Everything
```

## Quick Start

### Mode 1: Bring Your Own Scores

Use scores from DeepEval, Ragas, human reviewers, or any source.

```python
from faultmap import SliceAnalyzer

analyzer = SliceAnalyzer(model="gpt-4o-mini")

report = analyzer.analyze(
    prompts=prompts,
    responses=responses,
    scores=scores,  # list of floats in [0, 1]
)
print(report)
```

### Mode 2: Reference-Based Scoring

Provide ground-truth answers. faultmap scores via embedding cosine similarity.

```python
report = analyzer.analyze(
    prompts=prompts,
    responses=responses,
    references=ground_truth_answers,
)
```

### Mode 3: Reference-Free (Fully Autonomous)

No ground truth needed. faultmap estimates reliability via **semantic entropy**
and **self-consistency** by sampling multiple responses from the LLM.

```python
report = analyzer.analyze(
    prompts=prompts,
    responses=responses,
    # No scores, no references — faultmap scores autonomously
)
```

### Coverage Auditing

Find what your test suite is missing.

```python
coverage = analyzer.audit_coverage(
    test_prompts=my_test_set,
    production_prompts=my_production_logs,
)
print(coverage)
```

## Reading Results

```python
print(report.slices[0].name)           # "Legal compliance questions"
print(report.slices[0].failure_rate)   # 0.72
print(report.slices[0].baseline_rate)  # 0.15
print(report.slices[0].effect_size)    # 4.8
print(report.slices[0].adjusted_p_value)  # 0.0003
print(report.slices[0].sample_indices) # [12, 45, 67, ...]
print(report.slices[0].examples)       # [{"prompt": ..., "response": ..., "score": ...}]
```

## API Reference

### `SliceAnalyzer`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"gpt-4o-mini"` | litellm model string for LLM calls |
| `embedding_model` | `"all-MiniLM-L6-v2"` | Embedding model (local or API) |
| `significance_level` | `0.05` | Alpha for BH correction |
| `min_slice_size` | `10` | Minimum cluster size |
| `failure_threshold` | `0.5` | Score below this = failure |
| `n_samples` | `8` | Responses per prompt (Mode 3) |
| `clustering_method` | `"hdbscan"` | `"hdbscan"` or `"agglomerative"` |
| `max_concurrent_requests` | `50` | Max parallel LLM calls |
| `temperature` | `1.0` | Sampling temperature (Mode 3) |

### Methods

- `analyze(prompts, responses, scores=None, references=None) → AnalysisReport`
- `audit_coverage(test_prompts, production_prompts) → CoverageReport`

## License

Apache 2.0
```

---

## 2. Example Scripts

### `examples/example_mode1_custom_scores.py`

```python
"""
Example: Mode 1 — Bring Your Own Scores

Use pre-computed scores from any evaluation framework.
faultmap discovers which input slices have elevated failure rates.
"""
from faultmap import SliceAnalyzer

# Simulated evaluation data
prompts = (
    [f"What are the legal requirements for {topic}?"
     for topic in ["GDPR", "HIPAA", "SOC2", "PCI-DSS", "CCPA"] * 6]
    + [f"How do I reset {item}?"
       for item in ["password", "2FA", "email", "phone", "settings"] * 6]
    + [f"What is the price of {product}?"
       for product in ["basic", "pro", "enterprise", "team", "starter"] * 6]
)
responses = [f"Response for: {p}" for p in prompts]

# Legal questions fail, others pass
scores = [0.2] * 30 + [0.85] * 30 + [0.9] * 30

analyzer = SliceAnalyzer(
    model="gpt-4o-mini",
    embedding_model="all-MiniLM-L6-v2",
    min_slice_size=10,
)

report = analyzer.analyze(prompts, responses, scores=scores)
print(report)

for s in report.slices:
    print(f"\n{s.name}: {s.failure_rate:.0%} failure rate "
          f"(vs {s.baseline_rate:.0%} baseline, p={s.adjusted_p_value:.4f})")
```

### `examples/example_mode2_reference_based.py`

```python
"""
Example: Mode 2 — Reference-Based Scoring

Provide ground-truth answers. faultmap scores responses
using cosine similarity between response and reference embeddings.
"""
from faultmap import SliceAnalyzer

prompts = [
    "What is the capital of France?",
    "What is photosynthesis?",
    "Explain quantum computing",
    # ... more prompts
]
responses = [
    "Paris is the capital of France.",
    "Photosynthesis is how plants make food from sunlight.",
    "I'm not sure about quantum computing.",  # bad response
    # ...
]
references = [
    "The capital of France is Paris.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "Quantum computing uses qubits that can exist in superposition.",
    # ...
]

analyzer = SliceAnalyzer(model="gpt-4o-mini")
report = analyzer.analyze(prompts, responses, references=references)
print(report)
```

### `examples/example_mode3_reference_free.py`

```python
"""
Example: Mode 3 — Reference-Free (Fully Autonomous)

No ground truth needed. faultmap estimates reliability by sampling
multiple responses and measuring semantic entropy + self-consistency.

High entropy = model gives different answers each time = uncertain.
Low entropy = model is consistent = likely reliable.
"""
from faultmap import SliceAnalyzer

prompts = [
    "What is the capital of France?",       # factual, should be consistent
    "What will Bitcoin cost in 2030?",       # speculative, should be inconsistent
    "Explain the water cycle",              # factual
    "What is the meaning of life?",          # philosophical, inconsistent
    # ... more prompts
]
responses = [
    "Paris is the capital of France.",
    "Bitcoin will reach $100,000.",
    "Water evaporates, forms clouds, and falls as rain.",
    "The meaning of life is subjective.",
]

analyzer = SliceAnalyzer(
    model="gpt-4o-mini",
    n_samples=8,
    temperature=1.0,
)

report = analyzer.analyze(prompts, responses)
print(report)
print(f"\nScoring metadata: {report.metadata.get('scoring_metadata', {}).keys()}")
```

### `examples/example_coverage_audit.py`

```python
"""
Example: Coverage Auditing

Compare your test suite against production traffic to find
semantic regions your tests never cover.
"""
from faultmap import SliceAnalyzer

# Your test prompts
test_prompts = [
    "How do I reset my password?",
    "How do I change my email?",
    "How do I update payment info?",
    # ... your eval dataset
] * 10

# Production logs (from your app)
production_prompts = [
    "How do I reset my password?",
    "How do I change my email?",
    "How do I set up two-factor authentication?",  # not in test set!
    "How do I configure SSO for my organization?",  # not in test set!
    # ... real user queries
] * 10

analyzer = SliceAnalyzer(model="gpt-4o-mini")
coverage = analyzer.audit_coverage(test_prompts, production_prompts)
print(coverage)

for gap in coverage.gaps:
    print(f"\nGap: {gap.name}")
    print(f"  {gap.size} production prompts with no nearby test case")
    print(f"  Mean distance: {gap.mean_distance:.4f}")
    print(f"  Examples: {gap.representative_prompts[:3]}")
```

---

## 3. Packaging Verification Checklist

```bash
# 1. Clean install test
pip install -e .
python -c "from faultmap import SliceAnalyzer; print('OK')"

# 2. Optional deps install
pip install -e ".[local]"
python -c "from faultmap.embeddings import LocalEmbedder; print('OK')"

pip install -e ".[rich]"
python -c "from rich.console import Console; print('OK')"

# 3. Full test suite
pip install -e ".[all]"
pytest tests/ -v --cov=faultmap

# 4. Verify package metadata
pip show faultmap

# 5. Build distribution
pip install build
python -m build
ls dist/
```

---

## 4. Docstring Checklist

All public API must have docstrings by ship day:

- [ ] `SliceAnalyzer.__init__`
- [ ] `SliceAnalyzer.analyze`
- [ ] `SliceAnalyzer.audit_coverage`
- [ ] `AnalysisReport` (class + `summary`, `to_dict`)
- [ ] `CoverageReport` (class + `summary`, `to_dict`)
- [ ] `FailureSlice` (class)
- [ ] `CoverageGap` (class)
- [ ] `ScoringResult` (class)
- [ ] All exception classes (1-line docstrings already present)

---

## 5. Pre-Ship Final Review

Before declaring MVP complete:

1. **Functional**: All 3 scoring modes produce correct results
2. **Statistical**: BH correction controls FDR at stated alpha
3. **Edge cases**: Empty input, zero failures, single cluster, small dataset
4. **Error messages**: Clear and actionable (especially missing deps)
5. **`print(report)`**: Produces readable, useful output
6. **`report.to_dict()`**: Returns valid JSON-serializable dict
7. **No secrets**: No API keys, no hardcoded credentials
8. **Dependencies**: `pip install faultmap` works on a clean env
9. **Tests**: `pytest tests/ -v` passes 100%
10. **Logging**: `logging.getLogger("faultmap")` emits useful debug info

---

## Post-MVP Roadmap (Not Implemented, Architected For)

| Feature | Design hook |
|---------|-------------|
| `compare()` method | `analyzer.py` has clean async pattern to follow |
| Temporal tracking | `AnalysisReport.to_dict()` enables JSON serialization across runs |
| Export slices as datasets | `FailureSlice.sample_indices` enables reconstruction |
| Async public API | Internal pipeline is already async; expose `analyze_async()` |
| Caching | Embedder interface allows wrapping with cache layer |
| DeepEval/Ragas adapters | `PrecomputedScorer` accepts any float list |
| Web dashboard | `AnalysisReport.to_dict()` is JSON-ready |
