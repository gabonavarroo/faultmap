# faultmap

> Automatically discover where and why your LLM is failing.

[![PyPI version](https://img.shields.io/pypi/v/faultmap.svg)](https://pypi.org/project/faultmap/)
[![CI](https://github.com/gabonavarroo/faultmap/actions/workflows/ci.yml/badge.svg)](https://github.com/gabonavarroo/faultmap/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gabonavarroo/faultmap/blob/main/notebooks/tutorial.ipynb)

---

## The Problem

Your eval says **85% accuracy**. Users are complaining. Where are the failures?

Aggregate metrics hide critical patterns. A model can score well overall while catastrophically failing on specific input types — legal questions, billing disputes, technical setup — that matter most to your users.

**faultmap** is the diagnostic layer that answers two questions:

1. **"Where exactly is my model failing?"** — Automated discovery of input slices where failure rate is statistically significantly higher than baseline.
2. **"Can I trust my test suite?"** — Embedding-space coverage analysis that finds semantic blind spots your test suite has never touched.

---

## How It Works

```
Prompts → Embed → Cluster → Statistical Test → BH Correction → Name → Report
```

faultmap embeds your prompts into a semantic space, clusters them to find coherent input groups, runs hypothesis tests to identify clusters with elevated failure rates, applies Benjamini-Hochberg correction to control false discovery rate, and uses an LLM to generate human-readable names for each failure slice.

### Failure Slice Detection

1. **Embed** — your prompts are embedded into a high-dimensional vector space using a local or API-based embedding model
2. **Cluster** — HDBSCAN (default) or agglomerative clustering groups semantically similar prompts
3. **Test** — chi-squared or Fisher exact test compares each cluster's failure rate against the baseline
4. **Correct** — Benjamini-Hochberg FDR correction filters out false positives across all clusters
5. **Name** — an LLM automatically generates a human-readable name and description for each significant cluster

### Coverage Auditing

1. **Embed** — both your test prompts and production prompts are embedded
2. **Distance** — nearest-neighbor search finds production prompts that have no nearby test prompt
3. **Cluster** — uncovered production prompts are grouped into semantically coherent gap clusters
4. **Name** — the LLM names each gap so you know exactly what your test suite is missing

---

## Installation

```bash
pip install faultmap                # Core (uses API embeddings via litellm)
pip install faultmap[local]         # + sentence-transformers for local embeddings
pip install faultmap[rich]          # + rich for pretty terminal output
pip install faultmap[all]           # Everything
```

---

## Tutorial

An interactive Jupyter notebook walks through all four usage modes with a **mock path** (no API key needed) and equivalent real API code:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gabonavarroo/faultmap/blob/main/notebooks/tutorial.ipynb)

See [`notebooks/tutorial.ipynb`](notebooks/tutorial.ipynb).

---

## Quick Start

### Mode 1: Bring Your Own Scores

Use scores from DeepEval, Ragas, human reviewers, or any evaluation framework.

```python
from faultmap import SliceAnalyzer

analyzer = SliceAnalyzer(model="gpt-4o-mini")

report = analyzer.analyze(
    prompts=prompts,
    responses=responses,
    scores=scores,  # list of floats in [0, 1], lower = worse
)
print(report)
```

### Mode 2: Reference-Based Scoring

Provide ground-truth answers. faultmap scores responses via cosine similarity between response and reference embeddings.

```python
report = analyzer.analyze(
    prompts=prompts,
    responses=responses,
    references=ground_truth_answers,
)
```

### Mode 3: Reference-Free (Fully Autonomous)

No ground truth needed. faultmap estimates reliability via **semantic entropy** and **self-consistency** by sampling multiple responses from the LLM and measuring how consistent they are.

```python
report = analyzer.analyze(
    prompts=prompts,
    responses=responses,
    # No scores, no references — faultmap scores autonomously
)
```

### Coverage Auditing

Find what your test suite is missing by comparing it against production traffic.

```python
coverage = analyzer.audit_coverage(
    test_prompts=my_test_set,
    production_prompts=my_production_logs,
)
print(coverage)
```

---

## Reading Results

### Analysis Report

```python
# Top-level stats
print(report.total_prompts)          # 500
print(report.total_failures)         # 87
print(report.baseline_failure_rate)  # 0.174
print(report.num_significant)        # 2

# Inspect each failure slice
for s in report.slices:
    print(s.name)             # "Legal compliance questions"
    print(s.description)      # "Prompts asking about regulatory requirements..."
    print(s.size)             # 45
    print(s.failure_rate)     # 0.72
    print(s.baseline_rate)    # 0.17
    print(s.effect_size)      # 4.2   (how many times worse than baseline)
    print(s.adjusted_p_value) # 0.0003
    print(s.test_used)        # "chi2"

    # Recover the original data
    print(s.sample_indices)   # [12, 45, 67, ...]  — indices into your prompts list
    print(s.examples)         # [{"prompt": "...", "response": "...", "score": 0.1}, ...]
    print(s.representative_prompts)  # ["How do I comply with GDPR?", ...]

# Export to dict/JSON
import json
json.dumps(report.to_dict())
```

### Coverage Report

```python
print(coverage.overall_coverage_score)  # 0.82 (82% of prod prompts are covered)
print(coverage.num_gaps)                # 3
print(coverage.metadata["num_uncovered_total"])  # uncovered prompts, clustered or not
print(coverage.metadata["unclustered_prompt_indices"])  # uncovered prompts below gap threshold

for gap in coverage.gaps:
    print(gap.name)                    # "Two-factor authentication setup"
    print(gap.description)             # "Users asking about 2FA configuration..."
    print(gap.size)                    # 28 (production prompts with no test coverage)
    print(gap.mean_distance)           # 1.73 (avg distance to nearest test prompt)
    print(gap.representative_prompts)  # ["How do I enable 2FA?", ...]
    print(gap.prompt_indices)          # indices into your production_prompts list
```

---

## API Reference

### `SliceAnalyzer`

```python
analyzer = SliceAnalyzer(
    model="gpt-4o-mini",           # litellm model string for LLM calls (naming + Mode 3)
    embedding_model="text-embedding-3-small",  # default API embedding model
    significance_level=0.05,       # alpha for Benjamini-Hochberg FDR correction
    min_slice_size=10,             # minimum cluster size (smaller clusters ignored)
    failure_threshold=0.5,         # score below this is a failure (for binary classification)
    n_samples=8,                   # Mode 3 only: responses sampled per prompt
    clustering_method="hdbscan",   # "hdbscan" (auto k) or "agglomerative" (grid search)
    max_concurrent_requests=50,    # max parallel LLM API calls
    temperature=1.0,               # Mode 3 only: sampling temperature
    consistency_threshold=0.8,     # Mode 3 only: cosine sim threshold for self-consistency
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"gpt-4o-mini"` | litellm model string. Supports 100+ providers: `"anthropic/claude-3-haiku"`, `"ollama/mistral"`, etc. |
| `embedding_model` | `"text-embedding-3-small"` | API model string by default, so `pip install faultmap` works without extras. Local sentence-transformers models require `pip install faultmap[local]`. |
| `significance_level` | `0.05` | FDR alpha. Slices with `adjusted_p_value < significance_level` are reported. |
| `min_slice_size` | `10` | Minimum prompts per cluster. Smaller clusters are discarded before testing. |
| `failure_threshold` | `0.5` | Score cutoff. A prompt with `score < failure_threshold` is counted as a failure. |
| `n_samples` | `8` | Mode 3 only: number of additional responses sampled per prompt. Higher = more accurate entropy estimate. |
| `clustering_method` | `"hdbscan"` | `"hdbscan"` auto-discovers the number of clusters. `"agglomerative"` performs grid search over `[5, 10, 15, 20, 25, 30]` clusters. |
| `max_concurrent_requests` | `50` | Semaphore-limited concurrency for LLM calls. Reduce if hitting rate limits. |
| `temperature` | `1.0` | Mode 3 only: higher temperature increases response diversity for entropy estimation. |
| `consistency_threshold` | `0.8` | Mode 3 only: cosine similarity cutoff for self-consistency scoring. |

### `analyze()`

```python
report: AnalysisReport = analyzer.analyze(
    prompts: list[str],
    responses: list[str],
    scores: list[float] | None = None,      # Mode 1: pre-computed scores in [0, 1]
    references: list[str] | None = None,    # Mode 2: ground-truth answers
    # Neither → Mode 3 (entropy, autonomous)
    # Both → Mode 1 wins, Mode 2 is ignored (with warning)
)
```

### `audit_coverage()`

```python
report: CoverageReport = analyzer.audit_coverage(
    test_prompts: list[str],
    production_prompts: list[str],
    distance_threshold: float | None = None,  # auto-computed if None
    min_gap_size: int = 5,                    # minimum prompts to form a gap
)
```

---

## Scoring Modes In Depth

### Mode 1 — Precomputed Scores

Pass any float list where **higher = better quality**. The `failure_threshold` parameter (default `0.5`) decides the binary split.

Works with scores from any eval framework:
- [DeepEval](https://github.com/confident-ai/deepeval): `metric.score`
- [Ragas](https://github.com/explodinggradients/ragas): `result['answer_relevancy']`
- Human annotation: `1.0` (good) / `0.0` (bad)
- LLM-as-judge: normalized score in `[0, 1]`

### Mode 2 — Reference-Based

Provide `references` (ground-truth answers). faultmap:
1. Embeds both `responses` and `references` using the configured embedding model
2. Computes cosine similarity between each response and its reference
3. Uses similarity as the quality score

Best for: RAG evaluation, Q&A benchmarks, translation quality.

### Mode 3 — Entropy / Autonomous

No external labels needed. faultmap:
1. Samples `n_samples` additional responses per prompt at `temperature=1.0`
2. Embeds all samples + original response
3. Measures **semantic entropy**: how spread out are the responses in embedding space? (high entropy = inconsistent = uncertain)
4. Measures **self-consistency**: what fraction of samples are cosine-similar to the original?
5. Combines: `score = 0.5 * (1 - normalized_entropy) + 0.5 * self_consistency`

High entropy clusters reveal where the model is uncertain or hallucinating.
Best for: discovering unknown unknowns when you have no ground truth.

---

## Embedding Models

### Local (no API calls, requires `pip install faultmap[local]`)

```python
# Sentence-transformers models — auto-detected by prefix
SliceAnalyzer(embedding_model="all-MiniLM-L6-v2")    # fast, 384-dim
SliceAnalyzer(embedding_model="all-mpnet-base-v2")   # accurate, 768-dim
SliceAnalyzer(embedding_model="paraphrase-multilingual-MiniLM-L12-v2")  # multilingual
```

### API (via litellm)

```python
SliceAnalyzer()                                          # defaults to text-embedding-3-small
SliceAnalyzer(embedding_model="text-embedding-3-small")   # OpenAI
SliceAnalyzer(embedding_model="text-embedding-3-large")   # OpenAI
SliceAnalyzer(embedding_model="voyage/voyage-2")          # Voyage AI
```

For asymmetric embedding APIs, you can pass role-specific request options:

```python
SliceAnalyzer(
    embedding_model="nvidia_nim/nvidia/nv-embedqa-e5-v5",
    embedding_usage_kwargs={"query": {"input_type": "query"}},
)
```

API embeddings also truncate long texts to 2000 characters by default to avoid
strict provider token limits. Set `embedding_max_text_chars=None` to disable it.

---

## LLM Providers

faultmap uses [litellm](https://github.com/BerriAI/litellm) internally, so any provider litellm supports works:

```python
SliceAnalyzer(model="gpt-4o-mini")                      # OpenAI
SliceAnalyzer(model="anthropic/claude-3-haiku-20240307") # Anthropic
SliceAnalyzer(model="ollama/mistral")                    # Local via Ollama
SliceAnalyzer(model="groq/llama3-8b-8192")              # Groq
SliceAnalyzer(model="azure/gpt-4o-mini")                # Azure OpenAI
```

---

## Statistical Details

faultmap uses **one-sided hypothesis testing** — it only flags clusters that fail *more* than the baseline, never less.

**Test selection** (per cluster):
- If expected cell count < 5: Fisher exact test (exact, handles small samples)
- Otherwise: chi-squared with Yates continuity correction

**Multiple comparison correction**: Benjamini-Hochberg FDR at the configured `significance_level`. This controls the *false discovery rate* — the fraction of reported slices that are false positives — which is more appropriate than Bonferroni for exploration.

**No external statistics libraries**: chi-squared p-values are computed via `math.erfc(sqrt(chi2/2))` (exact for df=1), Fisher exact via `math.lgamma` to avoid overflow. Zero scipy/statsmodels dependency.

---

## Logging

```python
import logging
logging.getLogger("faultmap").setLevel(logging.DEBUG)
```

faultmap logs progress at each pipeline step: scoring mode, failure counts, cluster sizes, significance results.

---

## Contributing

```bash
git clone https://github.com/gabonavarroo/faultmap.git
cd faultmap
pip install -e ".[dev]"
pytest tests/ -v
ruff check .
```

All tests use mocks — no API keys needed to run the test suite.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.

---

## Examples

See the [`examples/`](examples/) directory:

- [`example_mode1_custom_scores.py`](examples/example_mode1_custom_scores.py) — Mode 1 with pre-computed scores
- [`example_mode2_reference_based.py`](examples/example_mode2_reference_based.py) — Mode 2 with ground-truth references
- [`example_mode3_reference_free.py`](examples/example_mode3_reference_free.py) — Mode 3 autonomous entropy scoring
- [`example_coverage_audit.py`](examples/example_coverage_audit.py) — Coverage gap detection

---

## License

Apache 2.0
