from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import pytest

from faultmap.embeddings import Embedder
from faultmap.llm import AsyncLLMClient

# ── Mock Embedder ──────────────────────────────────────────


class MockEmbedder(Embedder):
    """
    Deterministic embedder for testing.
    Maps text → embedding via hashing. No model downloads.
    """
    DIM = 64

    def embed(self, texts: list[str], *, usage: str = "generic") -> np.ndarray:
        if not texts:
            return np.empty((0, self.DIM), dtype=np.float32)
        embs = []
        for t in texts:
            seed = hash((t, usage)) % (2**31)
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
        "Name: Test Cluster\nDescription: A test cluster of similar prompts.\n"
        "Root Cause: The model lacks domain knowledge.\n"
        "Suggested Fix: Add domain context to the system prompt."
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


# ── Comparison Test Data ──────────────────────────────────


def make_comparison_data(
    n_clusters: int = 3,
    n_per_cluster: int = 30,
    dim: int = 64,
    a_better_clusters: list[int] | None = None,
    b_better_clusters: list[int] | None = None,
    a_fail_score: float = 0.2,
    a_pass_score: float = 0.8,
    b_fail_score: float = 0.2,
    b_pass_score: float = 0.8,
    seed: int = 42,
) -> dict:
    """Generate synthetic comparison data with known per-cluster winners.

    Creates well-separated clusters in embedding space and assigns scores so
    that each model has a clear advantage in designated clusters:

    - ``a_better_clusters``: Model A passes (``a_pass_score``), Model B fails
      (``b_fail_score``).
    - ``b_better_clusters``: Model A fails (``a_fail_score``), Model B passes
      (``b_pass_score``).
    - Remaining clusters: both models pass (``a_pass_score``, ``b_pass_score``).

    Default configuration (3 clusters × 30 prompts = 90 total):

    - Cluster 0: A wins (A=0.8, B=0.2)
    - Cluster 1: B wins (A=0.2, B=0.8)
    - Cluster 2: tied   (A=0.8, B=0.8)

    Args:
        n_clusters: Number of semantic clusters.
        n_per_cluster: Prompts per cluster.
        dim: Embedding dimension.
        a_better_clusters: Cluster indices where A is better. Default ``[0]``.
        b_better_clusters: Cluster indices where B is better. Default ``[1]``.
        a_fail_score: Score assigned to Model A in clusters where A is worse.
        a_pass_score: Score assigned to Model A in clusters where A is better
            or tied.
        b_fail_score: Score assigned to Model B in clusters where B is worse.
        b_pass_score: Score assigned to Model B in clusters where B is better
            or tied.
        seed: Random seed for reproducibility.

    Returns:
        dict with keys:
            prompts: list[str]
            responses_a: list[str]
            responses_b: list[str]
            scores_a: list[float]
            scores_b: list[float]
            embeddings: np.ndarray shape (n_total, dim)
            labels: np.ndarray shape (n_total,) — ground-truth cluster labels
            a_better_clusters: list[int]
            b_better_clusters: list[int]
    """
    if a_better_clusters is None:
        a_better_clusters = [0]
    if b_better_clusters is None:
        b_better_clusters = [1]

    rng = np.random.default_rng(seed)

    # Generate well-separated cluster centers on the unit sphere
    centers = rng.standard_normal((n_clusters, dim))
    for i in range(n_clusters):
        centers[i] = centers[i] / np.linalg.norm(centers[i]) * 5.0

    embeddings = []
    prompts = []
    responses_a = []
    responses_b = []
    scores_a = []
    scores_b = []
    labels = []

    cluster_topics = [
        "legal compliance", "billing disputes", "technical setup",
        "general questions", "product feedback", "account management",
    ]

    for c in range(n_clusters):
        topic = cluster_topics[c % len(cluster_topics)]
        cluster_embs = centers[c] + rng.standard_normal((n_per_cluster, dim)) * 0.15
        embeddings.append(cluster_embs)

        for j in range(n_per_cluster):
            prompts.append(
                f"[{topic}] prompt {c}-{j}: How do I handle {topic} issue #{j}?"
            )
            responses_a.append(f"Model A response about {topic} for query {j}")
            responses_b.append(f"Model B response about {topic} for query {j}")

            if c in a_better_clusters:
                # A passes, B fails
                scores_a.append(a_pass_score)
                scores_b.append(b_fail_score)
            elif c in b_better_clusters:
                # A fails, B passes
                scores_a.append(a_fail_score)
                scores_b.append(b_pass_score)
            else:
                # Tied — both pass
                scores_a.append(a_pass_score)
                scores_b.append(b_pass_score)

            labels.append(c)

    return {
        "prompts": prompts,
        "responses_a": responses_a,
        "responses_b": responses_b,
        "scores_a": scores_a,
        "scores_b": scores_b,
        "embeddings": np.vstack(embeddings).astype(np.float32),
        "labels": np.array(labels),
        "a_better_clusters": a_better_clusters,
        "b_better_clusters": b_better_clusters,
    }


@pytest.fixture
def comparison_data():
    """Default comparison data: 3 clusters × 30 prompts.

    Cluster 0: A wins (A=0.8, B=0.2).
    Cluster 1: B wins (A=0.2, B=0.8).
    Cluster 2: tied  (A=0.8, B=0.8).
    """
    return make_comparison_data()
