from __future__ import annotations
import logging

import numpy as np

from ..exceptions import ClusteringError

logger = logging.getLogger(__name__)


def cluster_embeddings(
    embeddings: np.ndarray,
    method: str = "hdbscan",
    min_cluster_size: int = 10,
    n_clusters: int | None = None,
) -> np.ndarray:
    """
    Cluster embedding vectors.

    Args:
        embeddings: (n, d) array.
        method: "hdbscan" or "agglomerative".
        min_cluster_size: Minimum cluster size.
        n_clusters: For agglomerative only. If None, auto-select via silhouette.

    Returns:
        labels: (n,) integer array. -1 = noise/removed.

    Key insight: L2-normalize before clustering. On unit vectors,
    euclidean distance is monotonically related to cosine distance:
        ||a - b||² = 2 - 2·cos(a, b)
    This lets us use euclidean metric (required by Ward linkage)
    while effectively clustering by cosine similarity.
    """
    n = embeddings.shape[0]

    if n < min_cluster_size:
        raise ClusteringError(
            f"Dataset has {n} prompts but min_cluster_size={min_cluster_size}. "
            f"Reduce min_cluster_size or provide more data."
        )

    if n < 30:
        logger.warning(
            f"Small dataset ({n} prompts). Clustering results may be unreliable."
        )

    # L2-normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    normed = embeddings / norms

    if method == "hdbscan":
        return _cluster_hdbscan(normed, min_cluster_size)
    elif method == "agglomerative":
        return _cluster_agglomerative(normed, min_cluster_size, n_clusters)
    else:
        raise ClusteringError(
            f"Unknown clustering method: {method!r}. Use 'hdbscan' or 'agglomerative'."
        )


def _cluster_hdbscan(normed: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """
    HDBSCAN clustering.

    Uses sklearn.cluster.HDBSCAN (available since sklearn 1.3).
    - cluster_selection_method="eom" (Excess of Mass) is the default
      and works well for finding clusters of varying density.
    - n_jobs=-1 for parallel distance computation.

    Error if all points are noise (no clusters found).
    """
    from sklearn.cluster import HDBSCAN

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        n_jobs=-1,
    )
    labels = clusterer.fit_predict(normed)

    unique_labels = set(labels)
    unique_labels.discard(-1)
    if not unique_labels:
        raise ClusteringError(
            "HDBSCAN found no clusters (all points classified as noise). "
            "Try reducing min_cluster_size or min_slice_size."
        )

    logger.info(
        f"HDBSCAN found {len(unique_labels)} clusters, "
        f"{np.sum(labels == -1)} noise points"
    )
    return labels


def _cluster_agglomerative(
    normed: np.ndarray,
    min_cluster_size: int,
    n_clusters: int | None,
) -> np.ndarray:
    """
    Agglomerative clustering with Ward linkage.

    If n_clusters is None, auto-select via silhouette score over
    candidates [5, 10, 15, 20, 25, 30] (filtered to < n // 2).

    Post-filter: set labels to -1 for clusters smaller than min_cluster_size.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    n = normed.shape[0]

    if n_clusters is None:
        candidates = [k for k in [5, 10, 15, 20, 25, 30] if k < n // 2]
        if not candidates:
            n_clusters = max(2, n // 5)
        else:
            best_k = candidates[0]
            best_score = -1.0
            for k in candidates:
                try:
                    ac = AgglomerativeClustering(n_clusters=k, linkage="ward")
                    temp_labels = ac.fit_predict(normed)
                    s = silhouette_score(
                        normed, temp_labels, metric="euclidean",
                        sample_size=min(n, 5000),
                    )
                    if s > best_score:
                        best_score = s
                        best_k = k
                except Exception:
                    continue
            n_clusters = best_k

    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = clusterer.fit_predict(normed)

    # Post-filter: remove small clusters
    unique, counts = np.unique(labels, return_counts=True)
    small_clusters = set(unique[counts < min_cluster_size])
    if small_clusters:
        mask = np.isin(labels, list(small_clusters))
        labels = labels.copy()
        labels[mask] = -1

    remaining = set(np.unique(labels))
    remaining.discard(-1)
    if not remaining:
        raise ClusteringError(
            f"All agglomerative clusters were smaller than "
            f"min_cluster_size={min_cluster_size}. Try reducing min_cluster_size."
        )

    logger.info(
        f"Agglomerative: {len(remaining)} clusters kept "
        f"(n_clusters param={n_clusters})"
    )
    return labels


def get_representative_prompts(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cluster_id: int,
    prompts: list[str],
    top_k: int = 5,
) -> tuple[list[str], list[int]]:
    """
    Get the top_k prompts closest to the cluster centroid.

    Algorithm:
    1. Get member indices where labels == cluster_id
    2. Compute centroid = mean of member embeddings
    3. Cosine similarity of each member to centroid
    4. Return top_k by similarity

    Returns:
        (representative_prompts, global_indices)
    """
    member_mask = labels == cluster_id
    member_indices = np.where(member_mask)[0]
    member_embs = embeddings[member_mask]

    centroid = member_embs.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid) + 1e-10
    centroid = centroid / centroid_norm

    emb_norms = np.linalg.norm(member_embs, axis=1, keepdims=True) + 1e-10
    normed_embs = member_embs / emb_norms

    sims = normed_embs @ centroid
    top_local = np.argsort(sims)[::-1][:top_k]

    top_global_indices = member_indices[top_local]
    top_prompts = [prompts[idx] for idx in top_global_indices]

    return top_prompts, top_global_indices.tolist()
