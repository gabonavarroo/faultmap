from unittest.mock import MagicMock

import numpy as np
import pytest

from faultmap.scoring.reference import ReferenceScorer


class TestReferenceScorer:
    def _make_mock_embedder(self, response_embs, reference_embs):
        """Create a mock embedder that returns known embeddings."""
        embedder = MagicMock()
        embedder.embed = MagicMock(side_effect=[
            np.array(response_embs, dtype=np.float32),
            np.array(reference_embs, dtype=np.float32),
        ])
        return embedder

    @pytest.mark.asyncio
    async def test_identical_texts_score_near_one(self):
        """Identical embeddings → cosine sim = 1.0 → score = 1.0."""
        emb = [[1.0, 0.0, 0.0]]
        embedder = self._make_mock_embedder(emb, emb)
        scorer = ReferenceScorer(embedder, ["ref"])
        result = await scorer.score(["prompt"], ["response"])
        assert np.isclose(result.scores[0], 1.0, atol=0.01)

    @pytest.mark.asyncio
    async def test_orthogonal_score_half(self):
        """Orthogonal embeddings → cosine sim = 0 → score = 0.5."""
        resp_emb = [[1.0, 0.0]]
        ref_emb = [[0.0, 1.0]]
        embedder = self._make_mock_embedder(resp_emb, ref_emb)
        scorer = ReferenceScorer(embedder, ["ref"])
        result = await scorer.score(["prompt"], ["response"])
        assert np.isclose(result.scores[0], 0.5, atol=0.01)

    @pytest.mark.asyncio
    async def test_opposite_score_near_zero(self):
        """Antiparallel embeddings → cosine sim = -1 → score = 0.0."""
        resp_emb = [[1.0, 0.0]]
        ref_emb = [[-1.0, 0.0]]
        embedder = self._make_mock_embedder(resp_emb, ref_emb)
        scorer = ReferenceScorer(embedder, ["ref"])
        result = await scorer.score(["prompt"], ["response"])
        assert np.isclose(result.scores[0], 0.0, atol=0.01)

    @pytest.mark.asyncio
    async def test_mode_is_reference(self):
        emb = [[1.0, 0.0]]
        embedder = self._make_mock_embedder(emb, emb)
        scorer = ReferenceScorer(embedder, ["ref"])
        result = await scorer.score(["p"], ["r"])
        assert result.mode == "reference"
