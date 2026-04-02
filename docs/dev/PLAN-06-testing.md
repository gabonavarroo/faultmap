# Phase 6 — Testing Strategy (Day 6)

**Goal**: Complete test suite, synthetic data infrastructure, and manual end-to-end validation.

**Files to create/update**:
- `tests/conftest.py` — shared fixtures, mock embedder, mock LLM, synthetic data generators
- Any missing `__init__.py` files under `tests/`

**Milestone**: `pytest tests/ -v` passes 100%. Manual e2e test with real APIs confirms the pipeline works.

---

## 1. `tests/conftest.py` — Shared Test Infrastructure

This is the most important test file. It provides reusable fixtures that all test modules depend on.

```python
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

from faultmap.embeddings import Embedder
from faultmap.llm import AsyncLLMClient


# ── Mock Embedder ──────────────────────────────────────────


class MockEmbedder(Embedder):
    """
    Deterministic embedder for testing.
    Maps text → embedding via hashing. No model downloads.
    """
    DIM = 64

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.DIM), dtype=np.float32)
        embs = []
        for t in texts:
            seed = hash(t) % (2**31)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.DIM)
            vec = vec / (np.linalg.norm(vec) + 1e-10)
            embs.append(vec)
        return np.array(embs, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self.DIM


@pytest.fixture
def mock_embedder():
    return MockEmbedder()


# ── Mock LLM Client ───────────────────────────────────────


@pytest.fixture
def mock_llm_client():
    """AsyncLLMClient mock that returns canned naming responses."""
    client = AsyncMock(spec=AsyncLLMClient)
    client.complete.return_value = (
        "Name: Test Cluster\nDescription: A test cluster of similar prompts."
    )
    client.complete_batch.return_value = []
    return client


# ── Synthetic Clustered Data ──────────────────────────────


def make_clustered_data(
    n_clusters: int = 3,
    n_per_cluster: int = 30,
    dim: int = 64,
    failure_clusters: list[int] | None = None,
    failure_score: float = 0.2,
    pass_score: float = 0.8,
    seed: int = 42,
) -> dict:
    """
    Generate synthetic evaluation data with known failure patterns.

    Creates well-separated clusters in embedding space. Specified clusters
    get low scores (failures), rest get high scores (passes).

    Args:
        n_clusters: Number of semantic clusters
        n_per_cluster: Prompts per cluster
        dim: Embedding dimension
        failure_clusters: Which cluster indices should fail (default: [0])
        failure_score: Score assigned to failure prompts
        pass_score: Score assigned to passing prompts
        seed: Random seed for reproducibility

    Returns:
        dict with keys:
            prompts: list[str]
            responses: list[str]
            scores: list[float]
            embeddings: np.ndarray (n_total, dim)
            labels: np.ndarray (n_total,) ground truth cluster labels
            failure_clusters: list[int]
    """
    if failure_clusters is None:
        failure_clusters = [0]

    rng = np.random.default_rng(seed)
    n_total = n_clusters * n_per_cluster

    # Generate cluster centers (well-separated on unit sphere)
    centers = rng.standard_normal((n_clusters, dim))
    for i in range(n_clusters):
        centers[i] = centers[i] / np.linalg.norm(centers[i]) * 5.0

    embeddings = []
    prompts = []
    responses = []
    scores = []
    labels = []

    cluster_topics = [
        "legal compliance", "billing disputes", "technical setup",
        "general questions", "product feedback", "account management",
    ]

    for c in range(n_clusters):
        topic = cluster_topics[c % len(cluster_topics)]
        is_failure = c in failure_clusters
        cluster_embs = centers[c] + rng.standard_normal((n_per_cluster, dim)) * 0.15
        embeddings.append(cluster_embs)

        for j in range(n_per_cluster):
            prompts.append(f"[{topic}] prompt {c}-{j}: How do I handle {topic} issue #{j}?")
            responses.append(f"Response about {topic} for query {j}")
            scores.append(failure_score if is_failure else pass_score)
            labels.append(c)

    return {
        "prompts": prompts,
        "responses": responses,
        "scores": scores,
        "embeddings": np.vstack(embeddings).astype(np.float32),
        "labels": np.array(labels),
        "failure_clusters": failure_clusters,
    }


@pytest.fixture
def clustered_data():
    """Default clustered data: 3 clusters, cluster 0 fails."""
    return make_clustered_data(n_clusters=3, n_per_cluster=30, failure_clusters=[0])


@pytest.fixture
def small_clustered_data():
    """Smaller dataset for faster tests."""
    return make_clustered_data(n_clusters=2, n_per_cluster=15, failure_clusters=[0])


# ── Coverage Test Data ────────────────────────────────────


def make_coverage_data(
    n_test: int = 50,
    n_prod_covered: int = 30,
    n_prod_gap: int = 20,
    dim: int = 64,
    seed: int = 42,
) -> dict:
    """
    Generate test + production data with a known coverage gap.

    Test prompts cluster in region A.
    Production prompts cluster in region A (covered) + region B (gap).

    Returns:
        dict with keys:
            test_prompts, test_embeddings,
            prod_prompts, prod_embeddings,
            gap_indices (indices of gap prompts in prod)
    """
    rng = np.random.default_rng(seed)

    center_a = rng.standard_normal(dim) * 5
    center_b = -center_a  # far away

    test_embs = center_a + rng.standard_normal((n_test, dim)) * 0.2
    prod_covered_embs = center_a + rng.standard_normal((n_prod_covered, dim)) * 0.2
    prod_gap_embs = center_b + rng.standard_normal((n_prod_gap, dim)) * 0.2

    prod_embs = np.vstack([prod_covered_embs, prod_gap_embs])
    n_prod = n_prod_covered + n_prod_gap

    return {
        "test_prompts": [f"test-prompt-{i}" for i in range(n_test)],
        "test_embeddings": test_embs.astype(np.float32),
        "prod_prompts": [f"prod-prompt-{i}" for i in range(n_prod)],
        "prod_embeddings": prod_embs.astype(np.float32),
        "gap_indices": list(range(n_prod_covered, n_prod)),
    }


@pytest.fixture
def coverage_data():
    return make_coverage_data()
```

---

## 2. Test Directory Structure

Ensure all `__init__.py` files exist:

```
tests/
├── __init__.py              # empty
├── conftest.py
├── test_utils.py            # from PLAN-01
├── test_embeddings.py       # from PLAN-02
├── test_llm.py              # from PLAN-02
├── test_labeling.py         # from PLAN-02
├── test_scoring/
│   ├── __init__.py          # empty
│   ├── test_precomputed.py  # from PLAN-03
│   ├── test_reference.py    # from PLAN-03
│   └── test_entropy.py      # from PLAN-03
├── test_slicing/
│   ├── __init__.py          # empty
│   ├── test_clustering.py   # from PLAN-04
│   └── test_statistics.py   # from PLAN-04
├── test_coverage/
│   ├── __init__.py          # empty
│   └── test_detector.py     # from PLAN-05
├── test_report.py           # from PLAN-05
└── test_analyzer.py         # from PLAN-05
```

---

## 3. Test Coverage Targets

| Module | Target | Critical paths |
|--------|--------|----------------|
| `utils.py` | 100% | Cosine similarity edge cases, validation |
| `embeddings.py` | 90% | Auto-detection, import error path, batching |
| `llm.py` | 90% | Retry logic, semaphore, batch ordering |
| `labeling.py` | 95% | Parsing well-formed + malformed responses |
| `scoring/precomputed.py` | 100% | Trivial |
| `scoring/reference.py` | 95% | Cosine mapping [-1,1]→[0,1] |
| `scoring/entropy.py` | 90% | Entropy math, consistency math, full pipeline |
| `slicing/clustering.py` | 85% | Both methods, edge cases, representatives |
| `slicing/statistics.py` | 95% | Chi2, Fisher, BH correction, monotonicity |
| `coverage/detector.py` | 90% | Gap detection, auto-threshold, clustering fallback |
| `report.py` | 80% | Plain text formatting (rich tested if installed) |
| `analyzer.py` | 85% | Mode detection, full pipeline, edge cases |

---

## 4. Manual End-to-End Test (Day 6)

After all unit tests pass, run a manual test with real APIs to confirm the full pipeline works.

### Mode 1 e2e test script

```python
"""Manual e2e test: Mode 1 with synthetic data."""
from faultmap import SliceAnalyzer

# Generate data with a known failure pattern
prompts = (
    [f"How do I comply with regulation {i}?" for i in range(30)]  # legal (fail)
    + [f"How do I reset my password for account {i}?" for i in range(30)]  # account (pass)
    + [f"What is the status of order {i}?" for i in range(30)]  # order (pass)
)
responses = [f"Response for: {p}" for p in prompts]
scores = [0.2] * 30 + [0.8] * 30 + [0.9] * 30  # legal cluster fails

analyzer = SliceAnalyzer(
    model="gpt-4o-mini",
    embedding_model="all-MiniLM-L6-v2",  # requires [local]
    min_slice_size=10,
)

report = analyzer.analyze(prompts, responses, scores=scores)
print(report)

# Verify:
# - Should find ~1 significant slice around "legal/compliance" prompts
# - failure_rate for that slice should be ~100%
# - baseline should be ~33%
assert report.num_significant >= 1
assert report.slices[0].failure_rate > 0.8
print("\nMode 1 e2e: PASSED")
```

### Coverage e2e test script

```python
"""Manual e2e test: Coverage auditing."""
from faultmap import SliceAnalyzer

test_prompts = [f"How do I reset my password for account {i}?" for i in range(50)]
production_prompts = (
    [f"How do I reset my password for account {i}?" for i in range(30)]
    + [f"How do I set up two-factor authentication for {i}?" for i in range(20)]  # gap
)

analyzer = SliceAnalyzer(
    model="gpt-4o-mini",
    embedding_model="all-MiniLM-L6-v2",
)

coverage = analyzer.audit_coverage(test_prompts, production_prompts)
print(coverage)

# Verify:
# - Should find a gap around "two-factor authentication" prompts
# - coverage_score should be < 1.0
assert coverage.overall_coverage_score < 1.0
print("\nCoverage e2e: PASSED")
```

---

## 5. Running Tests

```bash
# Full suite with coverage
pytest tests/ -v --cov=faultmap --cov-report=term-missing

# Quick smoke test
pytest tests/test_utils.py tests/test_slicing/ -v

# Just the critical statistical tests
pytest tests/test_slicing/test_statistics.py -v

# Just the entropy scorer
pytest tests/test_scoring/test_entropy.py -v
```
