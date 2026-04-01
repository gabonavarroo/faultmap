import numpy as np
import pytest
from faultmap.coverage.detector import detect_coverage_gaps
from faultmap.exceptions import ConfigurationError


def _make_coverage_data(seed=42):
    """
    Test embeddings cluster in region A.
    Production embeddings cluster in region A (covered) + region B (gap).
    """
    rng = np.random.default_rng(seed)
    dim = 32

    # Region A: test + some production
    center_a = rng.standard_normal(dim) * 5
    test_embs = center_a + rng.standard_normal((50, dim)) * 0.2
    prod_covered = center_a + rng.standard_normal((30, dim)) * 0.2

    # Region B: only production (gap)
    center_b = -center_a  # far away
    prod_gap = center_b + rng.standard_normal((20, dim)) * 0.2

    prod_embs = np.vstack([prod_covered, prod_gap])
    prod_prompts = [f"prompt-{i}" for i in range(50)]

    return test_embs.astype(np.float32), prod_embs.astype(np.float32), prod_prompts


class TestDetectCoverageGaps:
    def test_finds_gap(self):
        test_embs, prod_embs, prompts = _make_coverage_data()
        # Use explicit threshold=1.0: covered points are ~0 distance, gap points ~2.0.
        # Auto-threshold (mean+1.5*std) can exceed 2.0 for bimodal distributions,
        # so we set a fixed midpoint threshold here.
        gap_labels, distances, threshold = detect_coverage_gaps(
            test_embs, prod_embs, prompts, distance_threshold=1.0, min_gap_size=5,
        )
        # Should find gap in region B (indices 30-49)
        has_gap = np.any(gap_labels[30:] >= 0)
        assert has_gap, "Should detect a gap in the uncovered region"

    def test_all_covered(self):
        rng = np.random.default_rng(42)
        dim = 16
        # float64 avoids float32 rounding errors that produce tiny non-zero NN distances
        embs = rng.standard_normal((50, dim)).astype(np.float64)
        gap_labels, _, _ = detect_coverage_gaps(
            embs, embs, [f"p-{i}" for i in range(50)],
            min_gap_size=5,
        )
        # All should be -1 (covered) since test == prod
        assert np.all(gap_labels == -1)

    def test_empty_test_raises(self):
        with pytest.raises(ConfigurationError):
            detect_coverage_gaps(
                np.empty((0, 16)), np.random.randn(10, 16).astype(np.float32),
                [f"p-{i}" for i in range(10)],
            )

    def test_empty_prod_returns_empty(self):
        gap_labels, distances, _ = detect_coverage_gaps(
            np.random.randn(10, 16).astype(np.float32),
            np.empty((0, 16), dtype=np.float32),
            [],
        )
        assert len(gap_labels) == 0

    def test_returns_distances_array(self):
        test_embs, prod_embs, prompts = _make_coverage_data()
        gap_labels, distances, threshold = detect_coverage_gaps(
            test_embs, prod_embs, prompts, min_gap_size=5,
        )
        assert len(distances) == len(prod_embs)
        assert np.all(distances >= 0)

    def test_custom_threshold(self):
        test_embs, prod_embs, prompts = _make_coverage_data()
        # Very high threshold → nothing uncovered
        gap_labels, _, used_threshold = detect_coverage_gaps(
            test_embs, prod_embs, prompts,
            distance_threshold=1000.0, min_gap_size=5,
        )
        assert used_threshold == 1000.0
        assert np.all(gap_labels == -1)
