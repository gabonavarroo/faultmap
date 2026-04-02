import numpy as np
import pytest
from faultmap.slicing.clustering import (
    cluster_embeddings, get_representative_prompts,
)
from faultmap.exceptions import ClusteringError


def _make_clustered_embeddings(n_per_cluster=30, dim=64, n_clusters=3, seed=42):
    """Generate well-separated Gaussian clusters in high-d space."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim))
    # Ensure centers are far apart by normalizing and scaling
    for i in range(n_clusters):
        centers[i] = centers[i] / np.linalg.norm(centers[i]) * 5

    embeddings = []
    for center in centers:
        cluster = center + rng.standard_normal((n_per_cluster, dim)) * 0.1
        embeddings.append(cluster)

    return np.vstack(embeddings)


class TestClusterEmbeddings:
    def test_hdbscan_finds_clusters(self):
        embs = _make_clustered_embeddings(n_per_cluster=30, n_clusters=3)
        labels = cluster_embeddings(embs, method="hdbscan", min_cluster_size=10)
        unique = set(labels)
        unique.discard(-1)
        assert len(unique) >= 2  # Should find at least 2 of the 3 clusters

    def test_agglomerative_finds_clusters(self):
        embs = _make_clustered_embeddings(n_per_cluster=30, n_clusters=3)
        labels = cluster_embeddings(embs, method="agglomerative", min_cluster_size=5)
        unique = set(labels)
        unique.discard(-1)
        assert len(unique) >= 2

    def test_too_few_points_raises(self):
        embs = np.random.randn(3, 64)
        with pytest.raises(ClusteringError, match="min_cluster_size"):
            cluster_embeddings(embs, min_cluster_size=10)

    def test_unknown_method_raises(self):
        embs = np.random.randn(50, 64)
        with pytest.raises(ClusteringError, match="Unknown"):
            cluster_embeddings(embs, method="kmeans", min_cluster_size=5)


class TestGetRepresentativePrompts:
    def test_returns_correct_count(self):
        n = 50
        dim = 16
        embs = np.random.randn(n, dim).astype(np.float32)
        labels = np.array([0] * 20 + [1] * 30)
        prompts = [f"prompt-{i}" for i in range(n)]

        rep_prompts, rep_indices = get_representative_prompts(
            embs, labels, cluster_id=0, prompts=prompts, top_k=5
        )
        assert len(rep_prompts) == 5
        assert len(rep_indices) == 5
        assert all(0 <= idx < 20 for idx in rep_indices)
