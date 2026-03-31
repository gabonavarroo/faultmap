# Phase 3 — Scoring Pipeline (Day 3)

**Goal**: Implement all three scoring modes. This is the input layer for the slicing engine.

**Files to create**:
- `faultmap/scoring/__init__.py`
- `faultmap/scoring/base.py`
- `faultmap/scoring/precomputed.py`
- `faultmap/scoring/reference.py`
- `faultmap/scoring/entropy.py`
- `tests/test_scoring/test_precomputed.py`
- `tests/test_scoring/test_reference.py`
- `tests/test_scoring/test_entropy.py`

**Milestone**: All three scoring modes work independently with mocked dependencies.

---

## 1. `scoring/base.py` — Abstract Base

```python
from __future__ import annotations
from abc import ABC, abstractmethod

from ..models import ScoringResult


class BaseScorer(ABC):
    """Abstract base for all scoring modes."""

    @abstractmethod
    async def score(
        self,
        prompts: list[str],
        responses: list[str],
        **kwargs,
    ) -> ScoringResult:
        """
        Compute scores for prompt-response pairs.
        Returns ScoringResult with scores in [0, 1] where higher = better.
        """
        ...
```

---

## 2. `scoring/precomputed.py` — Mode 1

Trivial passthrough. User provides their own scores.

```python
from __future__ import annotations
from ..models import ScoringResult
from .base import BaseScorer


class PrecomputedScorer(BaseScorer):
    """Mode 1: User-provided scores. Pure passthrough."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    async def score(self, prompts, responses, **kwargs) -> ScoringResult:
        return ScoringResult(scores=list(self._scores), mode="precomputed")
```

No edge cases — validation already done in `utils.validate_inputs()`.

---

## 3. `scoring/reference.py` — Mode 2

Score = cosine similarity between response embedding and reference embedding.

```python
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
        resp_emb = self._embedder.embed(responses)
        ref_emb = self._embedder.embed(self._references)

        sims = cosine_similarity_pairs(resp_emb, ref_emb)
        scores = np.clip((sims + 1.0) / 2.0, 0.0, 1.0)

        return ScoringResult(
            scores=scores.tolist(),
            mode="reference",
            metadata={"embedding_model": getattr(self._embedder, 'model_name', 'unknown')},
        )
```

---

## 4. `scoring/entropy.py` — Mode 3 (Most Complex)

Reference-free scoring via semantic entropy + self-consistency.

### Algorithm Overview

```
For each prompt:
  1. SAMPLE N responses from the LLM at high temperature
  2. EMBED all N samples + the original response
  3. SEMANTIC ENTROPY: cluster the N sample embeddings by semantic
     similarity, compute Shannon entropy of cluster distribution
  4. SELF-CONSISTENCY: fraction of samples that semantically agree
     with the original response
  5. COMBINE: score = 0.5 * (1 - normalized_entropy) + 0.5 * self_consistency

High score → model is confident and consistent → likely correct
Low score  → model is uncertain and inconsistent → likely failing
```

### Full Implementation

```python
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

        all_embeddings = self._embedder.embed(all_texts_to_embed)

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

        Why greedy instead of sklearn clustering?
        - N is tiny (8 by default), O(n²) is negligible
        - No dependency on clustering hyperparameters
        - Matches the "semantic equivalence class" concept from Kuhn et al.

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
```

### Entropy Scorer — Worked Example

```
Prompt: "What is the capital of France?"

Sample responses (n=4):
  r1: "The capital of France is Paris."
  r2: "Paris is the capital city of France."
  r3: "France's capital is Paris."
  r4: "The capital is Paris."

All 4 embed to similar vectors → cosine sim > 0.8 → 1 cluster

Cluster distribution: [4/4] = [1.0]
Entropy: -1.0 * log(1.0) = 0.0
Normalized entropy: 0.0 / log(4) = 0.0

Self-consistency: all 4 agree with original → 4/4 = 1.0

Score = 0.5 * (1 - 0) + 0.5 * 1.0 = 1.0  ← highly confident
```

```
Prompt: "What will Bitcoin be worth in 2030?"

Sample responses (n=4):
  r1: "Bitcoin will reach $500,000."
  r2: "Probably around $50,000."
  r3: "It could be worthless."
  r4: "I estimate $150,000 by 2030."

All 4 embed to different vectors → 4 clusters

Cluster distribution: [1/4, 1/4, 1/4, 1/4]
Entropy: -4 * (0.25 * log(0.25)) = log(4) ≈ 1.386
Normalized entropy: 1.386 / log(4) = 1.0

Self-consistency: maybe 0/4 or 1/4 agree → ~0.0

Score = 0.5 * (1 - 1.0) + 0.5 * 0.0 = 0.0  ← highly uncertain
```

---

## 5. `scoring/__init__.py`

```python
from .base import BaseScorer
from .entropy import EntropyScorer
from .precomputed import PrecomputedScorer
from .reference import ReferenceScorer

__all__ = ["BaseScorer", "PrecomputedScorer", "ReferenceScorer", "EntropyScorer"]
```

---

## 6. Day 3 Tests

### `tests/test_scoring/test_precomputed.py`

```python
import pytest
from faultmap.scoring.precomputed import PrecomputedScorer


class TestPrecomputedScorer:
    @pytest.mark.asyncio
    async def test_passthrough(self):
        scores = [0.1, 0.5, 0.9]
        scorer = PrecomputedScorer(scores)
        result = await scorer.score(["a", "b", "c"], ["x", "y", "z"])
        assert result.scores == [0.1, 0.5, 0.9]
        assert result.mode == "precomputed"

    @pytest.mark.asyncio
    async def test_does_not_mutate(self):
        original = [0.5, 0.6]
        scorer = PrecomputedScorer(original)
        result = await scorer.score(["a", "b"], ["x", "y"])
        result.scores  # should be a copy
        original[0] = 999
        assert result.scores[0] == 0.5
```

### `tests/test_scoring/test_reference.py`

```python
import numpy as np
import pytest
from unittest.mock import MagicMock
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
```

### `tests/test_scoring/test_entropy.py` — Critical Tests

```python
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock
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
```

---

## Verification

After completing Day 3:
```bash
pytest tests/test_scoring/ -v
```
