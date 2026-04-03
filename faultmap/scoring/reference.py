from __future__ import annotations

import numpy as np

from ..embeddings import Embedder
from ..models import ScoringResult
from ..utils import cosine_similarity_pairs
from .base import BaseScorer


class ReferenceScorer(BaseScorer):
    """
    Mode 2: Score = cosine_similarity(embed(response), embed(reference)).

    Algorithm:
    1. Embed all responses → (n, d)
    2. Embed all references → (n, d)
    3. Compute element-wise cosine similarity → (n,) in [-1, 1]
    4. Map to [0, 1] via (sim + 1) / 2
       -1 → 0.0 (opposite meaning)
        0 → 0.5 (unrelated)
       +1 → 1.0 (identical meaning)
    5. Return as ScoringResult

    Edge cases:
    - Zero-norm embedding (degenerate/empty text) → cosine_similarity_pairs
      returns ~0 due to epsilon, maps to 0.5 (neutral)
    - All references identical → valid, scores reflect response quality
    """

    def __init__(self, embedder: Embedder, references: list[str]) -> None:
        self._embedder = embedder
        self._references = references

    async def score(self, prompts, responses, **kwargs) -> ScoringResult:
        resp_emb = self._embedder.embed_documents(responses)
        ref_emb = self._embedder.embed_documents(self._references)

        sims = cosine_similarity_pairs(resp_emb, ref_emb)
        scores = np.clip((sims + 1.0) / 2.0, 0.0, 1.0)

        return ScoringResult(
            scores=scores.tolist(),
            mode="reference",
            metadata={"embedding_model": getattr(self._embedder, 'model_name', 'unknown')},
        )
