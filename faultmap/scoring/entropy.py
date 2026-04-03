from __future__ import annotations

import logging

import numpy as np

from ..embeddings import Embedder
from ..llm import AsyncLLMClient
from ..models import ScoringResult
from .base import BaseScorer

logger = logging.getLogger(__name__)


class EntropyScorer(BaseScorer):
    """
    Mode 3 (reference-free): Semantic entropy + self-consistency scoring.

    Based on:
    - Kuhn et al. 2023 "Semantic Uncertainty" (ICLR 2023) — semantic entropy
    - Wang et al. 2022 "Self-Consistency Improves Chain of Thought" — consistency

    Score interpretation: HIGH = confident/reliable, LOW = uncertain/hallucinating.
    """

    def __init__(
        self,
        client: AsyncLLMClient,
        embedder: Embedder,
        n_samples: int = 8,
        temperature: float = 1.0,
        consistency_threshold: float = 0.8,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self.n_samples = n_samples
        self.temperature = temperature
        self.consistency_threshold = consistency_threshold

    async def score(self, prompts, responses, **kwargs) -> ScoringResult:
        n = len(prompts)

        # ── Step 1: Sample multiple responses ──────────────────────
        # Build one message list per (prompt, sample) pair
        # Total LLM calls: n * n_samples
        all_messages = []
        for prompt in prompts:
            for _ in range(self.n_samples):
                all_messages.append([{"role": "user", "content": prompt}])

        logger.info(
            f"Entropy scorer: sampling {n * self.n_samples} responses "
            f"({n} prompts x {self.n_samples} samples)"
        )

        all_sampled = await self._client.complete_batch(
            all_messages,
            temperature=self.temperature,
            max_tokens=1024,
            desc="Sampling responses",
        )

        # Reshape: sampled_responses[i] = list of n_samples strings for prompt i
        sampled_responses: list[list[str]] = []
        for i in range(n):
            start = i * self.n_samples
            end = start + self.n_samples
            sampled_responses.append(all_sampled[start:end])

        # ── Step 2: Embed everything in one batch ──────────────────
        # Flatten all sampled responses, append original responses
        flat_samples = [resp for group in sampled_responses for resp in group]
        all_texts_to_embed = flat_samples + list(responses)

        all_embeddings = self._embedder.embed_documents(all_texts_to_embed)

        # Split back out
        sample_embeddings_flat = all_embeddings[: n * self.n_samples]
        orig_embeddings = all_embeddings[n * self.n_samples :]

        # Reshape sample embeddings: (n, n_samples, d)
        d = all_embeddings.shape[1]
        sample_embeddings = sample_embeddings_flat.reshape(n, self.n_samples, d)

        # ── Step 3 & 4: Compute entropy and consistency per prompt ──
        semantic_entropies = np.zeros(n)
        self_consistencies = np.zeros(n)

        for i in range(n):
            samples_emb = sample_embeddings[i]  # (n_samples, d)
            orig_emb = orig_embeddings[i]        # (d,)

            semantic_entropies[i] = self._compute_semantic_entropy(samples_emb)
            self_consistencies[i] = self._compute_self_consistency(
                orig_emb, samples_emb
            )

        # ── Step 5: Combine ────────────────────────────────────────
        max_entropy = np.log(self.n_samples) if self.n_samples > 1 else 1.0
        normalized_entropy = np.clip(semantic_entropies / max_entropy, 0.0, 1.0)

        scores = (1.0 - normalized_entropy) * 0.5 + self_consistencies * 0.5
        scores = np.clip(scores, 0.0, 1.0)

        return ScoringResult(
            scores=scores.tolist(),
            mode="entropy",
            metadata={
                "n_samples": self.n_samples,
                "temperature": self.temperature,
                "semantic_entropy": semantic_entropies.tolist(),
                "self_consistency": self_consistencies.tolist(),
                "normalized_entropy": normalized_entropy.tolist(),
                "scores": scores.tolist(),
            },
        )

    def _compute_semantic_entropy(self, samples_emb: np.ndarray) -> float:
        """
        Compute semantic entropy for a set of sample embeddings.

        Algorithm (greedy clustering):
        1. Compute pairwise cosine similarity matrix (n_samples x n_samples)
        2. Greedy single-pass clustering:
           a. Mark all samples as unassigned
           b. While unassigned samples remain:
              - Pick first unassigned as new cluster center
              - Assign all unassigned with sim >= threshold to this cluster
        3. Compute probabilities: p_k = |cluster_k| / n_samples
        4. Entropy H = -Σ p_k * log(p_k)

        Edge cases:
        - n_samples = 1 → entropy = 0
        - All identical → 1 cluster → entropy = 0
        - All different → n_samples clusters → entropy = log(n_samples)
        """
        n = samples_emb.shape[0]
        if n <= 1:
            return 0.0

        # Normalize
        norms = np.linalg.norm(samples_emb, axis=1, keepdims=True) + 1e-10
        normed = samples_emb / norms

        # Pairwise cosine similarity
        sim_matrix = normed @ normed.T

        # Greedy clustering
        assigned = np.full(n, -1, dtype=int)
        cluster_id = 0
        for i in range(n):
            if assigned[i] >= 0:
                continue
            assigned[i] = cluster_id
            for j in range(i + 1, n):
                if assigned[j] >= 0:
                    continue
                if sim_matrix[i, j] >= self.consistency_threshold:
                    assigned[j] = cluster_id
            cluster_id += 1

        # Compute entropy from cluster sizes
        cluster_sizes = np.bincount(assigned)
        probs = cluster_sizes / n
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))
        return float(entropy)

    def _compute_self_consistency(
        self,
        orig_emb: np.ndarray,
        samples_emb: np.ndarray,
    ) -> float:
        """
        Fraction of samples that agree with the original response.

        Algorithm:
        1. Cosine similarity between original and each sample
        2. Count fraction where sim >= consistency_threshold

        Edge cases:
        - Zero-norm original → return 0.0
        """
        orig_norm = np.linalg.norm(orig_emb)
        if orig_norm < 1e-10:
            return 0.0

        orig_normed = orig_emb / orig_norm
        sample_norms = np.linalg.norm(samples_emb, axis=1) + 1e-10
        samples_normed = samples_emb / sample_norms[:, np.newaxis]

        sims = samples_normed @ orig_normed
        agreement = np.mean(sims >= self.consistency_threshold)
        return float(agreement)
