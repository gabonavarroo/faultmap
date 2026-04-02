from __future__ import annotations
import logging

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..exceptions import ClusteringError, ConfigurationError
from ..slicing.clustering import cluster_embeddings, get_representative_prompts

logger = logging.getLogger(__name__)


def detect_coverage_gaps(
    test_embeddings: np.ndarray,
    prod_embeddings: np.ndarray,
    prod_prompts: list[str],
    distance_threshold: float | None = None,
    min_gap_size: int = 5,
    clustering_method: str = "hdbscan",
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Find production prompts far from any test prompt.

    Algorithm:
    1. L2-normalize both test and prod embeddings
    2. Fit NearestNeighbors(k=1) on test embeddings
    3. Query with prod embeddings → distances array
    4. Auto-threshold if None: mean + 1.5 * std of distances
    5. Uncovered = distance > threshold
    6. If enough uncovered: cluster them into coherent gap groups
    7. Map cluster labels back to full prod array

    Args:
        test_embeddings: (n_test, d)
        prod_embeddings: (n_prod, d)
        prod_prompts: original production prompt strings
        distance_threshold: custom threshold, or None for auto
        min_gap_size: minimum uncovered prompts to form a gap
        clustering_method: "hdbscan" or "agglomerative"

    Returns:
        gap_labels: (n_prod,) -1=covered, >=0=gap cluster id
        nn_distances: (n_prod,) nearest-neighbor distances
        distance_threshold: the threshold that was used

    Edge cases:
    - No test prompts → ConfigurationError
    - No prod prompts → return empty arrays
    - All covered → valid result (all -1)
    - Too few uncovered to cluster → treat all uncovered as one gap
    """
    if test_embeddings.shape[0] == 0:
        raise ConfigurationError("test_embeddings must be non-empty")
    if prod_embeddings.shape[0] == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=float),
            distance_threshold or 0.0,
        )

    # L2-normalize
    test_norms = np.linalg.norm(test_embeddings, axis=1, keepdims=True) + 1e-10
    test_normed = test_embeddings / test_norms
    prod_norms = np.linalg.norm(prod_embeddings, axis=1, keepdims=True) + 1e-10
    prod_normed = prod_embeddings / prod_norms

    # Nearest neighbor
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
    nn.fit(test_normed)
    distances, _ = nn.kneighbors(prod_normed)
    distances = distances.ravel()

    # Auto-threshold
    if distance_threshold is None:
        distance_threshold = float(np.mean(distances) + 1.5 * np.std(distances))
        logger.info(f"Auto-computed distance threshold: {distance_threshold:.4f}")

    # Find uncovered prompts
    uncovered_mask = distances > distance_threshold
    n_uncovered = int(np.sum(uncovered_mask))
    logger.info(f"{n_uncovered}/{len(distances)} production prompts are uncovered")

    # Initialize all as -1 (covered)
    gap_labels = np.full(len(distances), -1, dtype=int)

    if n_uncovered < min_gap_size:
        return gap_labels, distances, distance_threshold

    # Cluster the uncovered embeddings
    uncovered_embeddings = prod_embeddings[uncovered_mask]
    try:
        uncovered_cluster_labels = cluster_embeddings(
            uncovered_embeddings,
            method=clustering_method,
            min_cluster_size=min_gap_size,
        )
    except ClusteringError:
        # Clustering failed → treat all uncovered as one gap
        logger.warning(
            "Clustering of uncovered prompts failed. Treating all as one gap."
        )
        uncovered_cluster_labels = np.zeros(n_uncovered, dtype=int)

    # Map back to full array
    uncovered_indices = np.where(uncovered_mask)[0]
    for local_idx, global_idx in enumerate(uncovered_indices):
        gap_labels[global_idx] = uncovered_cluster_labels[local_idx]

    return gap_labels, distances, distance_threshold
