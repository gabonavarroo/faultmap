# Phase 4 — Slicing Pipeline (Day 4)

**Goal**: The core analytical engine — clustering, statistical testing, BH correction.

**Files to create**:
- `faultmap/slicing/__init__.py`
- `faultmap/slicing/clustering.py`
- `faultmap/slicing/statistics.py`
- `tests/test_slicing/test_clustering.py`
- `tests/test_slicing/test_statistics.py`

**Milestone**: Can cluster embeddings, run per-cluster hypothesis tests, apply multiple hypothesis correction, and get representative prompts.

---

## 1. `slicing/clustering.py`

Supports both HDBSCAN (default) and agglomerative clustering.

```python
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
```

---

## 2. `slicing/statistics.py`

Statistical tests for per-cluster failure rate significance + BH FDR correction.
**No scipy or statsmodels** — implemented from stdlib `math.erfc` and `math.lgamma`.

```python
from __future__ import annotations
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClusterTestResult:
    """Result of statistical test for one cluster."""
    cluster_id: int
    size: int
    failure_count: int
    failure_rate: float
    p_value: float
    test_used: str                # "chi2" | "fisher" | "none"
    adjusted_p_value: float = 1.0  # Set after BH correction


def test_cluster_failure_rate(
    cluster_failures: int,
    cluster_size: int,
    total_failures: int,
    total_size: int,
    cluster_id: int,
) -> ClusterTestResult:
    """
    Test if cluster's failure rate is significantly HIGHER than baseline.
    One-sided test.

    2x2 contingency table:
                        Failed    Passed
    In cluster:           a         b        | cluster_size
    Not in cluster:       c         d        | total_size - cluster_size
                        ----      ----
                      total_fail  total_pass  | total_size

    Decision: expected count < 5 in any cell → Fisher exact, else chi-squared.

    Edge cases:
    - failure_rate <= baseline → skip test, p = 1.0
    - cluster_size == total_size → p = 1.0 (cluster IS the whole dataset)
    - denominator == 0 → p = 1.0
    """
    a = cluster_failures
    b = cluster_size - cluster_failures
    c = total_failures - cluster_failures
    d = (total_size - cluster_size) - c

    failure_rate = a / cluster_size if cluster_size > 0 else 0.0
    baseline_rate = total_failures / total_size if total_size > 0 else 0.0

    # If cluster failure rate is not higher than baseline, skip
    if failure_rate <= baseline_rate:
        return ClusterTestResult(
            cluster_id=cluster_id,
            size=cluster_size,
            failure_count=a,
            failure_rate=failure_rate,
            p_value=1.0,
            test_used="none",
        )

    # Compute expected values to decide which test
    row_totals = [cluster_size, total_size - cluster_size]
    col_totals = [total_failures, total_size - total_failures]
    expected = np.array([
        [row_totals[0] * col_totals[0] / total_size,
         row_totals[0] * col_totals[1] / total_size],
        [row_totals[1] * col_totals[0] / total_size,
         row_totals[1] * col_totals[1] / total_size],
    ])

    if np.min(expected) < 5:
        p_value = _fisher_exact_one_sided(a, b, c, d)
        test_used = "fisher"
    else:
        p_value = _chi2_yates(a, b, c, d, total_size)
        test_used = "chi2"

    return ClusterTestResult(
        cluster_id=cluster_id,
        size=cluster_size,
        failure_count=a,
        failure_rate=failure_rate,
        p_value=p_value,
        test_used=test_used,
    )


def _chi2_yates(a: int, b: int, c: int, d: int, n: int) -> float:
    """
    Chi-squared with Yates' continuity correction for 2x2 table.

    Formula: chi2 = n * (|ad - bc| - n/2)² / ((a+b)(c+d)(a+c)(b+d))

    p-value for df=1: p = erfc(sqrt(chi2 / 2))
    This uses the exact relationship for chi2 with 1 degree of freedom.

    One-sided: divide two-sided p by 2 (we only care about higher rate).
    """
    from math import erfc, sqrt

    numerator = (abs(a * d - b * c) - n / 2.0) ** 2 * n
    denominator = (a + b) * (c + d) * (a + c) * (b + d)

    if denominator == 0:
        return 1.0

    chi2_stat = numerator / denominator
    p_two_sided = erfc(sqrt(chi2_stat / 2.0))
    return p_two_sided / 2.0


def _fisher_exact_one_sided(a: int, b: int, c: int, d: int) -> float:
    """
    Fisher exact test, one-sided (greater), without scipy.

    P(X >= a) where X ~ Hypergeometric(N=a+b+c+d, K=a+c, n=a+b).

    Uses log-gamma to avoid factorial overflow on large tables.
    """
    from math import lgamma, exp

    N = a + b + c + d
    K = a + c       # total failures
    n = a + b       # cluster size

    def log_choose(nn: int, kk: int) -> float:
        if kk < 0 or kk > nn:
            return float('-inf')
        return lgamma(nn + 1) - lgamma(kk + 1) - lgamma(nn - kk + 1)

    log_denom = log_choose(N, n)

    # Sum P(X = x) for x = a to min(n, K)
    max_x = min(n, K)
    p_value = 0.0
    for x in range(a, max_x + 1):
        log_p = log_choose(K, x) + log_choose(N - K, n - x) - log_denom
        p_value += exp(log_p)

    return min(p_value, 1.0)


def benjamini_hochberg(
    results: list[ClusterTestResult],
    alpha: float = 0.05,
) -> list[ClusterTestResult]:
    """
    Apply Benjamini-Hochberg FDR correction.

    Algorithm:
    1. Sort by p_value ascending
    2. adjusted[i] = p_value[i] * m / rank[i]  (rank is 1-indexed)
    3. Enforce monotonicity: backward pass, each = min(self, next)
    4. Clip to [0, 1]
    5. Assign to results

    Returns results sorted by adjusted_p_value ascending.

    Edge cases:
    - Empty list → return []
    - Single test → adjusted = raw p-value (m/rank = 1/1 = 1)
    - All p = 1.0 → all adjusted = 1.0
    """
    if not results:
        return []

    sorted_results = sorted(results, key=lambda r: r.p_value)
    m = len(sorted_results)

    # Raw adjusted p-values
    adjusted = [0.0] * m
    for i, result in enumerate(sorted_results):
        rank = i + 1
        adjusted[i] = result.p_value * m / rank

    # Enforce monotonicity (backward pass)
    for i in range(m - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    # Clip and assign
    for i, result in enumerate(sorted_results):
        result.adjusted_p_value = min(max(adjusted[i], 0.0), 1.0)

    return sorted(sorted_results, key=lambda r: r.adjusted_p_value)
```

### BH Correction — Worked Example

```
Input p-values: [0.001, 0.04, 0.03, 0.01, 0.06]
Sorted:         [0.001, 0.01, 0.03, 0.04, 0.06]
m = 5

Raw adjusted (p * m / rank):
  rank 1: 0.001 * 5 / 1 = 0.005
  rank 2: 0.01  * 5 / 2 = 0.025
  rank 3: 0.03  * 5 / 3 = 0.050
  rank 4: 0.04  * 5 / 4 = 0.050
  rank 5: 0.06  * 5 / 5 = 0.060

Monotonicity enforcement (backward pass):
  [0.005, 0.025, 0.050, 0.050, 0.060]  ← already monotone

At alpha = 0.05: first 4 are significant (adjusted_p ≤ 0.05)
```

---

## 3. `slicing/__init__.py`

```python
from .clustering import cluster_embeddings, get_representative_prompts
from .statistics import (
    ClusterTestResult,
    benjamini_hochberg,
    test_cluster_failure_rate,
)

__all__ = [
    "cluster_embeddings",
    "get_representative_prompts",
    "ClusterTestResult",
    "test_cluster_failure_rate",
    "benjamini_hochberg",
]
```

---

## 4. Day 4 Tests

### `tests/test_slicing/test_clustering.py`

```python
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
```

### `tests/test_slicing/test_statistics.py` — Critical

```python
import numpy as np
import pytest
from faultmap.slicing.statistics import (
    test_cluster_failure_rate,
    benjamini_hochberg,
    ClusterTestResult,
    _chi2_yates,
    _fisher_exact_one_sided,
)


class TestChi2Yates:
    def test_known_table(self):
        """
        Table:  Failed  Passed
        Cluster:  40      10     | 50
        Other:    40      110    | 150
                  80      120    | 200

        Expected failure rate in cluster: 40/50 = 80%
        Expected failure rate overall: 80/200 = 40%
        This should be highly significant.
        """
        p = _chi2_yates(a=40, b=10, c=40, d=110, n=200)
        assert p < 0.001  # Highly significant

    def test_no_difference(self):
        """Equal failure rates → p near 1.0."""
        # 20/50 in cluster vs 80/200 overall → both 40%
        p = _chi2_yates(a=20, b=30, c=60, d=90, n=200)
        assert p > 0.3  # Not significant


class TestFisherExact:
    def test_small_table_significant(self):
        """
        Table:  Failed  Passed
        Cluster:  7       1     | 8
        Other:    3       19    | 22
                  10      20    | 30
        """
        p = _fisher_exact_one_sided(a=7, b=1, c=3, d=19)
        assert p < 0.001

    def test_small_table_not_significant(self):
        """Proportional table → not significant."""
        # 3/8 in cluster vs 10/30 overall → ~same rate
        p = _fisher_exact_one_sided(a=3, b=5, c=7, d=15)
        assert p > 0.1


class TestClusterFailureRate:
    def test_higher_rate_uses_chi2(self):
        """Large enough expected counts → chi2."""
        result = test_cluster_failure_rate(
            cluster_failures=40, cluster_size=50,
            total_failures=80, total_size=200, cluster_id=0,
        )
        assert result.test_used == "chi2"
        assert result.p_value < 0.01
        assert result.failure_rate == 0.8

    def test_small_cluster_uses_fisher(self):
        """Small expected counts → Fisher."""
        result = test_cluster_failure_rate(
            cluster_failures=7, cluster_size=8,
            total_failures=10, total_size=30, cluster_id=0,
        )
        assert result.test_used == "fisher"
        assert result.p_value < 0.01

    def test_rate_not_higher_skips(self):
        """Cluster failure rate ≤ baseline → p = 1.0, test = 'none'."""
        result = test_cluster_failure_rate(
            cluster_failures=5, cluster_size=50,
            total_failures=80, total_size=200, cluster_id=0,
        )
        assert result.p_value == 1.0
        assert result.test_used == "none"


class TestBenjaminiHochberg:
    def test_known_correction(self):
        """Verify BH on known p-values."""
        results = [
            ClusterTestResult(i, 50, 10, 0.2, p, "chi2")
            for i, p in enumerate([0.001, 0.01, 0.03, 0.04, 0.06])
        ]
        corrected = benjamini_hochberg(results, alpha=0.05)

        adjusted = [r.adjusted_p_value for r in corrected]
        # First 4 should be ≤ 0.05, last should be > 0.05
        assert adjusted[0] <= 0.05   # 0.005
        assert adjusted[1] <= 0.05   # 0.025
        assert adjusted[2] <= 0.05   # 0.050
        assert adjusted[3] <= 0.05   # 0.050
        assert adjusted[4] > 0.05    # 0.060

    def test_empty(self):
        assert benjamini_hochberg([], alpha=0.05) == []

    def test_single_result(self):
        results = [ClusterTestResult(0, 50, 10, 0.2, 0.03, "chi2")]
        corrected = benjamini_hochberg(results, alpha=0.05)
        assert corrected[0].adjusted_p_value == 0.03  # m/rank = 1/1

    def test_monotonicity(self):
        """Adjusted p-values should be non-decreasing."""
        results = [
            ClusterTestResult(i, 50, 10, 0.2, p, "chi2")
            for i, p in enumerate([0.001, 0.005, 0.01, 0.05, 0.1, 0.5])
        ]
        corrected = benjamini_hochberg(results)
        adjusted = [r.adjusted_p_value for r in corrected]
        for i in range(len(adjusted) - 1):
            assert adjusted[i] <= adjusted[i + 1]
```

---

## Verification

After completing Day 4:
```bash
pytest tests/test_slicing/ tests/test_labeling.py -v
```
