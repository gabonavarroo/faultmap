from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from faultmap.scoring.entropy import EntropyScorer


@pytest.fixture
def mock_llm_client():
    return AsyncMock()


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    return embedder


class TestSemanticEntropy:
    """Test _compute_semantic_entropy directly with synthetic embeddings."""

    def _make_scorer(self):
        return EntropyScorer(
            client=AsyncMock(),
            embedder=MagicMock(),
            n_samples=8,
            consistency_threshold=0.8,
        )

    def test_all_identical_zero_entropy(self):
        """All samples identical → 1 cluster → entropy = 0."""
        scorer = self._make_scorer()
        embs = np.tile([1.0, 0.0, 0.0], (8, 1))  # 8 identical vectors
        entropy = scorer._compute_semantic_entropy(embs)
        assert entropy == 0.0

    def test_all_orthogonal_max_entropy(self):
        """All samples orthogonal → n clusters → entropy = log(n)."""
        scorer = self._make_scorer()
        n = 4
        scorer.n_samples = n
        embs = np.eye(n)  # 4 orthogonal vectors in 4-d space
        entropy = scorer._compute_semantic_entropy(embs)
        expected = np.log(n)
        assert np.isclose(entropy, expected, atol=0.01)

    def test_two_clusters(self):
        """4 samples in 2 clusters of 2 → entropy = log(2)."""
        scorer = self._make_scorer()
        scorer.consistency_threshold = 0.9
        # Two pairs of very similar vectors
        embs = np.array([
            [1.0, 0.0, 0.0],  # cluster 0
            [0.99, 0.01, 0.0],  # cluster 0
            [0.0, 1.0, 0.0],  # cluster 1
            [0.01, 0.99, 0.0],  # cluster 1
        ])
        entropy = scorer._compute_semantic_entropy(embs)
        expected = np.log(2)
        assert np.isclose(entropy, expected, atol=0.1)

    def test_single_sample(self):
        """Single sample → entropy = 0."""
        scorer = self._make_scorer()
        embs = np.array([[1.0, 0.0]])
        entropy = scorer._compute_semantic_entropy(embs)
        assert entropy == 0.0


class TestSelfConsistency:
    def _make_scorer(self):
        return EntropyScorer(
            client=AsyncMock(),
            embedder=MagicMock(),
            n_samples=4,
            consistency_threshold=0.8,
        )

    def test_all_agree(self):
        """All samples similar to original → consistency = 1.0."""
        scorer = self._make_scorer()
        orig = np.array([1.0, 0.0, 0.0])
        samples = np.array([
            [0.98, 0.02, 0.0],
            [0.95, 0.05, 0.0],
            [0.99, 0.01, 0.0],
            [0.97, 0.03, 0.0],
        ])
        consistency = scorer._compute_self_consistency(orig, samples)
        assert consistency == 1.0

    def test_none_agree(self):
        """All samples orthogonal to original → consistency = 0.0."""
        scorer = self._make_scorer()
        orig = np.array([1.0, 0.0, 0.0])
        samples = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, -1.0, 0.0],
        ])
        consistency = scorer._compute_self_consistency(orig, samples)
        assert consistency == 0.0

    def test_zero_norm_original(self):
        scorer = self._make_scorer()
        orig = np.zeros(3)
        samples = np.array([[1.0, 0.0, 0.0]])
        consistency = scorer._compute_self_consistency(orig, samples)
        assert consistency == 0.0


class TestEntropyFullPipeline:
    @pytest.mark.asyncio
    async def test_confident_prompt(self, mock_llm_client, mock_embedder):
        """All sampled responses identical → score near 1.0."""
        n_samples = 4

        # LLM returns the same response every time
        mock_llm_client.complete_batch.return_value = [
            "Paris is the capital" for _ in range(n_samples)
        ]

        # All embeddings are identical
        dim = 8
        identical_vec = np.random.randn(dim)
        identical_vec = identical_vec / np.linalg.norm(identical_vec)
        all_embs = np.tile(identical_vec, (n_samples + 1, 1)).astype(np.float32)
        mock_embedder.embed.return_value = all_embs

        scorer = EntropyScorer(
            client=mock_llm_client,
            embedder=mock_embedder,
            n_samples=n_samples,
            consistency_threshold=0.8,
        )

        result = await scorer.score(["What is the capital of France?"], ["Paris"])
        assert result.scores[0] > 0.8
        assert result.mode == "entropy"
        assert result.metadata["scores"] == result.scores

    @pytest.mark.asyncio
    async def test_uncertain_prompt(self, mock_llm_client, mock_embedder):
        """All sampled responses different → score near 0.0."""
        n_samples = 4

        mock_llm_client.complete_batch.return_value = [
            f"Response {i}" for i in range(n_samples)
        ]

        # Orthogonal embeddings for samples, plus original at end
        dim = n_samples + 1
        all_embs = np.eye(dim, dtype=np.float32)  # each vector is orthogonal
        mock_embedder.embed.return_value = all_embs

        scorer = EntropyScorer(
            client=mock_llm_client,
            embedder=mock_embedder,
            n_samples=n_samples,
            consistency_threshold=0.8,
        )

        result = await scorer.score(["Speculative question"], ["Guess"])
        assert result.scores[0] < 0.2
