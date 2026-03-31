# faultmap -- Complete Implementation Plan

## Table of Contents
1. [pyproject.toml and Dependencies](#1-pyprojecttoml-and-dependencies)
2. [Module: exceptions.py](#2-module-exceptionspy)
3. [Module: models.py](#3-module-modelspy)
4. [Module: utils.py](#4-module-utilspy)
5. [Module: llm.py](#5-module-llmpy)
6. [Module: embeddings.py](#6-module-embeddingspy)
7. [Module: labeling.py](#7-module-labelingpy)
8. [Module: scoring/](#8-module-scoring)
9. [Module: slicing/](#9-module-slicing)
10. [Module: coverage/](#10-module-coverage)
11. [Module: report.py](#11-module-reportpy)
12. [Module: analyzer.py](#12-module-analyzerpy)
13. [Module: __init__.py](#13-module-__init__py)
14. [Test Strategy](#14-test-strategy)
15. [Implementation Schedule](#15-implementation-schedule)

---

## 1. pyproject.toml and Dependencies

**File**: `/Users/gabo/Developer/Gabon/faultmap/pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "faultmap"
version = "0.1.0"
description = "Automatically discover where and why your LLM is failing"
readme = "README.md"
license = "Apache-2.0"
requires-python = ">=3.10"
authors = [{ name = "Gabon" }]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "numpy>=1.24",
    "scikit-learn>=1.3",
    "litellm>=1.30",
    "tqdm>=4.60",
    "nest-asyncio>=1.5",
]

[project.optional-dependencies]
local = ["sentence-transformers>=2.2"]
rich = ["rich>=13.0"]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "pytest-cov>=4.0",
    "ruff>=0.4",
    "mypy>=1.5",
]
all = ["faultmap[local,rich,dev]"]

[tool.ruff]
target-version = "py310"
line-length = 99

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
```

**Dependency rationale**:
- `numpy`: Array operations for embeddings, cosine similarity, entropy math. No way around it.
- `scikit-learn`: Provides `AgglomerativeClustering`, `NearestNeighbors`, cosine similarity utilities. HDBSCAN is available in sklearn >=1.3 via `sklearn.cluster.HDBSCAN`.
- `litellm`: LLM provider abstraction. Single dep for 100+ models.
- `tqdm`: Progress bars for async operations.
- `nest-asyncio`: Jupyter compatibility for `asyncio.run()`.
- `sentence-transformers` (optional `[local]`): Local embedding models.
- `rich` (optional `[rich]`): Pretty table formatting.

**Key decision**: Use `sklearn.cluster.HDBSCAN` (available since scikit-learn 1.3) instead of the standalone `hdbscan` package. This avoids a separate C-extension dependency and simplifies installation. The sklearn implementation covers our use case fully.

---

## 2. Module: exceptions.py

**File**: `/Users/gabo/Developer/Gabon/faultmap/faultmap/exceptions.py`

```python
class FaultmapError(Exception):
    """Base exception for all faultmap errors."""

class EmbeddingError(FaultmapError):
    """Raised when embedding computation fails."""

class ScoringError(FaultmapError):
    """Raised when scoring computation fails."""

class LLMError(FaultmapError):
    """Raised when LLM calls fail after retries."""

class ClusteringError(FaultmapError):
    """Raised when clustering produces degenerate results."""

class ConfigurationError(FaultmapError):
    """Raised for invalid parameter combinations."""
```

No complex logic. Straightforward hierarchy. All inherit from `FaultmapError` so users can catch broadly.

---

## 3. Module: models.py

**File**: `/Users/gabo/Developer/Gabon/faultmap/faultmap/models.py`

All data containers are `@dataclass(frozen=True)` for immutability. Using `__slots__` where possible.

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

@dataclass(frozen=True)
class ScoringResult:
    """Output of any scoring mode."""
    scores: list[float]                    # Per-prompt score in [0, 1]. Higher = better.
    mode: str                              # "precomputed" | "reference" | "entropy"
    metadata: dict = field(default_factory=dict)
    # metadata contents vary by mode:
    #   precomputed: {} (empty)
    #   reference: {"embedding_model": str}
    #   entropy: {"n_samples": int, "temperature": float,
    #             "semantic_entropy": list[float], "self_consistency": list[float]}


@dataclass(frozen=True)
class FailureSlice:
    """A single discovered failure cluster."""
    name: str                              # LLM-generated human-readable name
    description: str                       # LLM-generated 1-sentence description
    size: int                              # Number of prompts in this slice
    failure_rate: float                    # Fraction of prompts that failed in this slice
    baseline_failure_rate: float           # Global failure rate for comparison
    p_value: float                         # Raw p-value from statistical test
    adjusted_p_value: float                # BH-corrected p-value
    test_used: str                         # "chi2" | "fisher"
    representative_prompts: list[str]      # Top-k prompts closest to cluster centroid
    prompt_indices: list[int]              # Indices into original prompts list
    cluster_id: int                        # Internal cluster label


@dataclass(frozen=True)
class AnalysisReport:
    """Complete output of SliceAnalyzer.analyze()."""
    slices: list[FailureSlice]             # Significant slices, sorted by adjusted_p_value
    total_prompts: int
    total_failures: int
    baseline_failure_rate: float
    significance_level: float              # Alpha used
    failure_threshold: float               # Score threshold used
    scoring_mode: str                      # "precomputed" | "reference" | "entropy"
    num_clusters_tested: int               # Total clusters before filtering
    num_significant: int                   # len(slices)
    clustering_method: str                 # "hdbscan" | "agglomerative"
    embedding_model: str
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Return a plain-text one-paragraph summary."""
        if not self.slices:
            return (
                f"No statistically significant failure slices found among "
                f"{self.total_prompts} prompts (baseline failure rate: "
                f"{self.baseline_failure_rate:.1%}, alpha={self.significance_level})."
            )
        worst = self.slices[0]
        return (
            f"Found {self.num_significant} significant failure slice(s) among "
            f"{self.total_prompts} prompts. Worst slice: \"{worst.name}\" "
            f"({worst.size} prompts, {worst.failure_rate:.1%} failure rate vs "
            f"{self.baseline_failure_rate:.1%} baseline, adj. p={worst.adjusted_p_value:.4f})."
        )


@dataclass(frozen=True)
class CoverageGap:
    """A single discovered gap in test coverage."""
    name: str                              # LLM-generated name for the gap
    description: str                       # LLM-generated description
    size: int                              # Number of production prompts in this gap
    mean_distance: float                   # Average NN distance to nearest test prompt
    representative_prompts: list[str]      # Top-k production prompts in this gap
    prompt_indices: list[int]              # Indices into production_prompts list
    cluster_id: int


@dataclass(frozen=True)
class CoverageReport:
    """Complete output of SliceAnalyzer.audit_coverage()."""
    gaps: list[CoverageGap]                # Gaps sorted by mean_distance descending
    num_test_prompts: int
    num_production_prompts: int
    num_gaps: int
    overall_coverage_score: float          # 1 - (fraction of prod prompts in gaps)
    distance_threshold: float              # Used to define "uncovered"
    embedding_model: str
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Return a plain-text one-paragraph summary."""
        if not self.gaps:
            return (
                f"Test suite appears to cover production traffic well. "
                f"No significant gaps found among {self.num_production_prompts} "
                f"production prompts."
            )
        worst = self.gaps[0]
        return (
            f"Found {self.num_gaps} coverage gap(s). "
            f"Overall coverage score: {self.overall_coverage_score:.1%}. "
            f"Largest gap: \"{worst.name}\" ({worst.size} uncovered production prompts, "
            f"mean distance={worst.mean_distance:.3f})."
        )
```

**Edge cases**:
- `FailureSlice.representative_prompts` capped at 5 (configurable in clustering, hardcoded default).
- `AnalysisReport.slices` is always sorted by `adjusted_p_value` ascending (most significant first).
- `CoverageReport.gaps` is always sorted by `mean_distance` descending (most distant first).

---

## 4. Module: utils.py

**File**: `/Users/gabo/Developer/Gabon/faultmap/faultmap/utils.py`

```python
from __future__ import annotations
import asyncio
from typing import TypeVar, Callable, Awaitable

import numpy as np

T = TypeVar("T")


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity between rows of a and b.

    Args:
        a: shape (n, d)
        b: shape (m, d)

    Returns:
        shape (n, m) similarity matrix with values in [-1, 1].
    """
    # Normalize rows
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T


def cosine_similarity_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute element-wise cosine similarity between paired rows.

    Args:
        a: shape (n, d)
        b: shape (n, d)

    Returns:
        shape (n,) array of similarities.
    """
    dot = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1) + 1e-10
    norm_b = np.linalg.norm(b, axis=1) + 1e-10
    return dot / (norm_a * norm_b)


def run_sync(coro: Awaitable[T]) -> T:
    """
    Run an async coroutine synchronously.
    Handles Jupyter's already-running event loop via nest_asyncio.

    Algorithm:
    1. Try asyncio.get_running_loop()
    2. If loop exists: apply nest_asyncio and use loop.run_until_complete()
    3. If no loop: use asyncio.run()
    """
    import nest_asyncio
    try:
        loop = asyncio.get_running_loop()
        nest_asyncio.apply(loop)
        return loop.run_until_complete(coro)
    except RuntimeError:
        # No running loop
        nest_asyncio.apply()
        return asyncio.run(coro)


def validate_inputs(
    prompts: list[str],
    responses: list[str],
    scores: list[float] | None,
    references: list[str] | None,
) -> None:
    """
    Validate user inputs before analysis. Raises ConfigurationError.

    Checks:
    - prompts and responses are non-empty lists of equal length
    - All elements are strings
    - If scores provided: same length, all numeric, all in [0, 1] or convertible
    - If references provided: same length, all strings
    - scores and references are mutually exclusive (warn, not error -- scores takes priority)
    """
    from .exceptions import ConfigurationError

    if not prompts:
        raise ConfigurationError("prompts must be a non-empty list")
    if len(prompts) != len(responses):
        raise ConfigurationError(
            f"prompts ({len(prompts)}) and responses ({len(responses)}) must have equal length"
        )
    if scores is not None and len(scores) != len(prompts):
        raise ConfigurationError(
            f"scores ({len(scores)}) must have same length as prompts ({len(prompts)})"
        )
    if references is not None and len(references) != len(prompts):
        raise ConfigurationError(
            f"references ({len(references)}) must have same length as prompts ({len(prompts)})"
        )
    if scores is not None:
        for i, s in enumerate(scores):
            if not isinstance(s, (int, float)):
                raise ConfigurationError(f"scores[{i}] must be numeric, got {type(s).__name__}")
            if not (0.0 <= s <= 1.0):
                raise ConfigurationError(
                    f"scores[{i}]={s} is out of range [0, 1]. "
                    "Normalize scores to [0, 1] before passing to faultmap."
                )


def batch_items(items: list[T], batch_size: int) -> list[list[T]]:
    """Split a list into batches of at most batch_size."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
```

---

## 5. Module: llm.py

**File**: `/Users/gabo/Developer/Gabon/faultmap/faultmap/llm.py`

Central async LLM calling with rate limiting. Used by `labeling.py` and `scoring/entropy.py`.

```python
from __future__ import annotations
import asyncio
import logging
from typing import Any

from .exceptions import LLMError

logger = logging.getLogger(__name__)


class AsyncLLMClient:
    """
    Async wrapper around litellm.acompletion with semaphore-based rate limiting.
    """

    def __init__(
        self,
        model: str,
        max_concurrent_requests: int = 50,
        max_retries: int = 3,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> str:
        """
        Single completion call with retry + semaphore.

        Args:
            messages: OpenAI-format messages list.
            temperature: Sampling temperature.
            max_tokens: Max output tokens.

        Returns:
            The string content of the first choice.

        Raises:
            LLMError: After all retries exhausted.

        Algorithm:
        1. Acquire semaphore
        2. For attempt in range(max_retries):
            a. Call litellm.acompletion(model, messages, temperature, max_tokens, timeout)
            b. On success: return response.choices[0].message.content
            c. On rate limit (status 429): exponential backoff = 2^attempt seconds
            d. On other error: exponential backoff = 2^attempt seconds
        3. Raise LLMError with last exception
        """
        import litellm

        last_error: Exception | None = None
        async with self._semaphore:
            for attempt in range(self.max_retries):
                try:
                    response = await litellm.acompletion(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=self.timeout,
                        **kwargs,
                    )
                    content = response.choices[0].message.content
                    if content is None:
                        raise LLMError("LLM returned None content")
                    return content.strip()
                except Exception as e:
                    last_error = e
                    wait = 2 ** attempt
                    logger.warning(
                        f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {wait}s."
                    )
                    await asyncio.sleep(wait)

        raise LLMError(
            f"LLM call failed after {self.max_retries} retries: {last_error}"
        ) from last_error

    async def complete_batch(
        self,
        messages_list: list[list[dict[str, str]]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        desc: str = "LLM calls",
        show_progress: bool = True,
        **kwargs: Any,
    ) -> list[str]:
        """
        Batch completion with progress bar.

        Algorithm:
        1. Create tasks = [self.complete(msgs, ...) for msgs in messages_list]
        2. Use tqdm + asyncio.as_completed to show progress
        3. Collect results maintaining original order

        Returns:
            List of responses in same order as messages_list.
        """
        from tqdm.asyncio import tqdm_asyncio

        tasks = [
            self.complete(msgs, temperature=temperature, max_tokens=max_tokens, **kwargs)
            for msgs in messages_list
        ]

        if show_progress:
            results = await tqdm_asyncio.gather(*tasks, desc=desc)
        else:
            results = await asyncio.gather(*tasks)

        return list(results)
```

**Edge cases**:
- If litellm is not installed, import error at call time is fine (it is a required dependency).
- Semaphore is created per-instance; if user creates multiple `AsyncLLMClient`s, they have independent limits.
- `timeout` is per-call, not total.

---

## 6. Module: embeddings.py

**File**: `/Users/gabo/Developer/Gabon/faultmap/faultmap/embeddings.py`

```python
from __future__ import annotations
import logging
from abc import ABC, abstractmethod

import numpy as np

from .exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class Embedder(ABC):
    """Abstract base class for text embedding."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim).
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...


class LocalEmbedder(Embedder):
    """
    Embedding via sentence-transformers (local, no API calls).

    Constructor:
        model_name: str = "all-MiniLM-L6-v2"
        batch_size: int = 64
        device: str | None = None  (auto-detect: CUDA > MPS > CPU)

    Algorithm for embed():
    1. Import sentence_transformers. If ImportError: raise EmbeddingError with install instructions.
    2. Lazy-load model on first call (self._model is None initially).
    3. Call self._model.encode(texts, batch_size=self.batch_size, show_progress_bar=len(texts)>100)
    4. Return as np.ndarray (float32).

    Edge cases:
    - Empty list: return np.empty((0, self.dimension))
    - Single string accidentally passed: wrap in list
    """

    # List of known local model name patterns (not exhaustive, but covers the common ones).
    # Used by get_embedder() to decide Local vs API.
    LOCAL_MODEL_PREFIXES = (
        "all-MiniLM",
        "all-mpnet",
        "paraphrase-",
        "multi-qa-",
        "msmarco-",
        "sentence-t5",
        "e5-",
        "bge-",
        "gte-",
        "nomic-",
        "BAAI/",
        "intfloat/",
        "thenlper/",
    )

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._dimension: int | None = None

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise EmbeddingError(
                f"sentence-transformers is required for local embedding model "
                f"'{self.model_name}'. Install it with: pip install faultmap[local]"
            )
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            self._load_model()
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        return self._dimension  # type: ignore[return-value]


class APIEmbedder(Embedder):
    """
    Embedding via litellm's embedding API (OpenAI, Cohere, etc.).

    Constructor:
        model_name: str  (litellm embedding model string, e.g. "text-embedding-3-small")
        batch_size: int = 128  (API batch size per request)

    Algorithm for embed():
    1. Import litellm.
    2. Split texts into batches of self.batch_size.
    3. For each batch: call litellm.embedding(model=self.model_name, input=batch)
    4. Extract vectors from response.data[i].embedding.
    5. Concatenate into np.ndarray.

    Edge cases:
    - Empty list: return np.empty((0, self._dimension)) -- but we may not know dim yet.
      On first call with empty list, call with ["test"] to probe dimension, discard result.
    - Rate limiting: litellm handles retries internally for embedding calls.
    """

    def __init__(self, model_name: str, batch_size: int = 128) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._dimension: int | None = None

    def embed(self, texts: list[str]) -> np.ndarray:
        import litellm

        if not texts:
            if self._dimension is not None:
                return np.empty((0, self._dimension), dtype=np.float32)
            # Probe dimension
            probe = litellm.embedding(model=self.model_name, input=["dimension probe"])
            self._dimension = len(probe.data[0]["embedding"])
            return np.empty((0, self._dimension), dtype=np.float32)

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                response = litellm.embedding(model=self.model_name, input=batch)
                # Sort by index to maintain order
                sorted_data = sorted(response.data, key=lambda x: x["index"])
                all_embeddings.extend([d["embedding"] for d in sorted_data])
            except Exception as e:
                raise EmbeddingError(f"API embedding failed: {e}") from e

        result = np.array(all_embeddings, dtype=np.float32)
        self._dimension = result.shape[1]
        return result

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            import litellm
            probe = litellm.embedding(model=self.model_name, input=["dimension probe"])
            self._dimension = len(probe.data[0]["embedding"])
        return self._dimension


def get_embedder(model_name: str) -> Embedder:
    """
    Factory function: determine if model_name is local or API-based and return
    the appropriate Embedder.

    Decision logic:
    1. If model_name starts with any prefix in LocalEmbedder.LOCAL_MODEL_PREFIXES -> LocalEmbedder
    2. If model_name contains "/" and first part looks like a HuggingFace org -> LocalEmbedder
    3. Otherwise -> APIEmbedder

    This heuristic is imperfect. Users can always instantiate LocalEmbedder or APIEmbedder
    directly if auto-detection fails.
    """
    name_lower = model_name.lower()
    for prefix in LocalEmbedder.LOCAL_MODEL_PREFIXES:
        if name_lower.startswith(prefix.lower()):
            return LocalEmbedder(model_name)

    # Check for HuggingFace-style org/model names that are likely local
    if "/" in model_name:
        org = model_name.split("/")[0].lower()
        local_orgs = {"sentence-transformers", "baai", "intfloat", "thenlper", "nomic-ai"}
        if org in local_orgs:
            return LocalEmbedder(model_name)

    return APIEmbedder(model_name)
```

---

## 7. Module: labeling.py

**File**: `/Users/gabo/Developer/Gabon/faultmap/faultmap/labeling.py`

Shared naming module: given a cluster of texts, ask an LLM to produce a concise name and description. Used by both failure slice discovery and coverage gap detection.

```python
from __future__ import annotations
import asyncio
from dataclasses import dataclass

from .llm import AsyncLLMClient


@dataclass(frozen=True)
class ClusterLabel:
    """Name and description for a cluster."""
    name: str            # 2-5 word name, e.g. "Date Formatting Queries"
    description: str     # One sentence description


# Maximum number of representative texts to include in the naming prompt
MAX_EXAMPLES_FOR_NAMING = 15
# Maximum character length per example to avoid huge prompts
MAX_CHARS_PER_EXAMPLE = 300


async def label_cluster(
    client: AsyncLLMClient,
    representative_texts: list[str],
    context: str = "failure slice",
) -> ClusterLabel:
    """
    Ask the LLM to name a single cluster of texts.

    Args:
        client: The AsyncLLMClient to use.
        representative_texts: Sample texts from the cluster (ideally 5-15).
        context: Either "failure slice" or "coverage gap" -- changes the prompt framing.

    Algorithm:
    1. Truncate texts to MAX_EXAMPLES_FOR_NAMING, each truncated to MAX_CHARS_PER_EXAMPLE.
    2. Build a system prompt explaining the task.
    3. Build a user prompt with numbered examples.
    4. Call client.complete() with temperature=0.
    5. Parse response: expect "Name: <name>\nDescription: <description>".
    6. If parsing fails, use first line as name and rest as description.

    Returns:
        ClusterLabel with name and description.
    """
    truncated = [
        t[:MAX_CHARS_PER_EXAMPLE] + ("..." if len(t) > MAX_CHARS_PER_EXAMPLE else "")
        for t in representative_texts[:MAX_EXAMPLES_FOR_NAMING]
    ]

    examples_text = "\n".join(f"{i+1}. {t}" for i, t in enumerate(truncated))

    system_prompt = (
        f"You are analyzing a cluster of similar text inputs that form a {context}. "
        f"Given the example texts below, provide:\n"
        f"1. A concise name (2-5 words) that captures the common theme\n"
        f"2. A one-sentence description of what these texts have in common\n\n"
        f"Respond in exactly this format:\n"
        f"Name: <your name>\n"
        f"Description: <your description>"
    )

    user_prompt = f"Here are the example texts from this cluster:\n\n{examples_text}"

    response = await client.complete(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=150,
    )

    return _parse_label_response(response)


def _parse_label_response(response: str) -> ClusterLabel:
    """
    Parse "Name: ...\nDescription: ..." format.
    Fallback: first line = name, rest = description.
    """
    lines = response.strip().split("\n")
    name = ""
    description = ""

    for line in lines:
        line_stripped = line.strip()
        if line_stripped.lower().startswith("name:"):
            name = line_stripped[len("name:"):].strip()
        elif line_stripped.lower().startswith("description:"):
            description = line_stripped[len("description:"):].strip()

    if not name:
        name = lines[0].strip()[:80]
    if not description:
        description = " ".join(lines[1:]).strip()[:200] if len(lines) > 1 else name

    return ClusterLabel(name=name, description=description)


async def label_clusters(
    client: AsyncLLMClient,
    clusters_texts: list[list[str]],
    context: str = "failure slice",
) -> list[ClusterLabel]:
    """
    Label multiple clusters concurrently.

    Args:
        client: The AsyncLLMClient.
        clusters_texts: List of lists, where each inner list contains representative
                        texts for one cluster.
        context: "failure slice" or "coverage gap".

    Returns:
        List of ClusterLabels in same order as clusters_texts.
    """
    tasks = [
        label_cluster(client, texts, context=context)
        for texts in clusters_texts
    ]
    return list(await asyncio.gather(*tasks))
```

---

## 8. Module: scoring/

### 8a. scoring/base.py

**File**: `/Users/gabo/Developer/Gabon/faultmap/faultmap/scoring/base.py`

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

        Returns:
            ScoringResult with scores in [0, 1] where higher = better.
        """
        ...
```

### 8b. scoring/precomputed.py

**File**: `/Users/gabo/Developer/Gabon/faultmap/faultmap/scoring/precomputed.py`

```python
from __future__ import annotations

from ..models import ScoringResult
from .base import BaseScorer


class PrecomputedScorer(BaseScorer):
    """
    Mode 1: User-provided scores. Pure passthrough.

    Constructor:
        scores: list[float]  -- already validated in [0, 1] by utils.validate_inputs

    Algorithm for score():
    1. Return ScoringResult(scores=self.scores, mode="precomputed")

    No async work needed but we keep the interface consistent.
    """

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    async def score(
        self,
        prompts: list[str],
        responses: list[str],
        **kwargs,
    ) -> ScoringResult:
        return ScoringResult(scores=list(self._scores), mode="precomputed")
```

### 8c. scoring/reference.py

**File**: `/Users/gabo/Developer/Gabon/faultmap/faultmap/scoring/reference.py`

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

    Constructor:
        embedder: Embedder
        references: list[str]

    Algorithm for score():
    1. Embed responses: resp_emb = embedder.embed(responses)  # (n, d)
    2. Embed references: ref_emb = embedder.embed(references)  # (n, d)
    3. Compute pairwise cosine similarity: sims = cosine_similarity_pairs(resp_emb, ref_emb)
    4. Clamp to [0, 1]: scores = np.clip((sims + 1) / 2, 0, 1)
       NOTE: cosine similarity is in [-1, 1]. We map to [0, 1] via (sim + 1) / 2.
       This means: sim=-1 -> 0, sim=0 -> 0.5, sim=1 -> 1.0.
    5. Return ScoringResult(scores=scores.tolist(), mode="reference", metadata={...})

    Edge cases:
    - Zero-length embedding (degenerate): returns 0.5 (neutral).
    - All identical references: valid, will just cluster on response quality.
    """

    def __init__(self, embedder: Embedder, references: list[str]) -> None:
        self._embedder = embedder
        self._references = references

    async def score(
        self,
        prompts: list[str],
        responses: list[str],
        **kwargs,
    ) -> ScoringResult:
        resp_emb = self._embedder.embed(responses)
        ref_emb = self._embedder.embed(self._references)

        sims = cosine_similarity_pairs(resp_emb, ref_emb)
        # Map [-1, 1] -> [0, 1]
        scores = np.clip((sims + 1.0) / 2.0, 0.0, 1.0)

        return ScoringResult(
            scores=scores.tolist(),
            mode="reference",
            metadata={"embedding_model": str(self._embedder)},
        )
```

### 8d. scoring/entropy.py -- THE MOST COMPLEX PIECE

**File**: `/Users/gabo/Developer/Gabon/faultmap/faultmap/scoring/entropy.py`

```python
from __future__ import annotations
import asyncio
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

    When there is no ground truth, we estimate confidence by:
    1. Sampling N diverse responses from the LLM for each prompt
    2. Computing semantic entropy across the N samples
    3. Computing self-consistency (agreement fraction)
    4. Combining into a single confidence score

    Constructor:
        client: AsyncLLMClient  -- for sampling multiple responses
        embedder: Embedder      -- for computing semantic similarity between samples
        n_samples: int = 8      -- number of responses to sample per prompt
        temperature: float = 1.0  -- sampling temperature (higher = more diverse)
        consistency_threshold: float = 0.8  -- cosine sim threshold for "agreement"

    Score interpretation: HIGH score = HIGH confidence = model is CONSISTENT
    (and thus likely correct). LOW score = LOW confidence = model is UNCERTAIN
    (and thus likely failing). This is inverted from the other modes where
    low score = bad. We invert at the end so that low score = bad uniformly.
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

    async def score(
        self,
        prompts: list[str],
        responses: list[str],
        **kwargs,
    ) -> ScoringResult:
        """
        Full entropy scoring pipeline.

        Algorithm:
        1. SAMPLE: For each prompt, generate n_samples responses at high temperature.
           - Build messages: [{"role": "user", "content": prompt}] for each prompt
           - Total LLM calls: len(prompts) * n_samples
           - Use client.complete_batch() for all calls at once (semaphore handles rate limiting)
           - Result: sampled_responses[i][j] = j-th sample for i-th prompt

        2. EMBED: Embed all sampled responses + the original responses.
           - Flatten sampled_responses into a single list
           - Embed all at once for efficiency
           - Reshape back: sample_embeddings[i] = (n_samples, d) for prompt i
           - Also embed original responses: orig_embeddings = (n, d)

        3. SEMANTIC ENTROPY: For each prompt i:
           a. Compute pairwise cosine similarity matrix among the n_samples embeddings
              sim_matrix[j][k] = cosine_sim(sample_emb[j], sample_emb[k])
           b. Cluster samples into semantic equivalence classes:
              - Use greedy clustering: sort by similarity to first sample
              - Two samples are "equivalent" if cosine_sim > consistency_threshold
              - This gives K clusters (1 <= K <= n_samples)
           c. Compute class probabilities: p_k = |class_k| / n_samples
           d. Semantic entropy = -sum(p_k * log(p_k)) for k in 1..K
           e. Normalize: normalized_entropy = entropy / log(n_samples)
              (so it is in [0, 1], where 0 = all samples identical, 1 = all different)

        4. SELF-CONSISTENCY: For each prompt i:
           a. Compute cosine_sim(original_response_emb, each_sample_emb)
           b. Fraction of samples that agree with original: those with sim > threshold
           c. self_consistency[i] = agreement_fraction (in [0, 1])

        5. COMBINE: Final score = (1 - normalized_entropy) * 0.5 + self_consistency * 0.5
           - Both components are in [0, 1]
           - High score = low entropy + high self-consistency = confident/reliable
           - Low score = high entropy + low self-consistency = uncertain/unreliable
           - Equal weighting is a sensible default; could be configurable later.

        6. RETURN: ScoringResult with combined scores and metadata including
           per-prompt entropy and consistency values.
        """
        n = len(prompts)

        # Step 1: Sample multiple responses
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

        # Step 2: Embed everything
        # Flatten all sampled responses
        flat_samples = [resp for group in sampled_responses for resp in group]
        all_texts_to_embed = flat_samples + responses  # samples first, then originals

        all_embeddings = self._embedder.embed(all_texts_to_embed)

        # Split back out
        sample_embeddings_flat = all_embeddings[: n * self.n_samples]  # (n*n_samples, d)
        orig_embeddings = all_embeddings[n * self.n_samples :]          # (n, d)

        # Reshape sample embeddings: (n, n_samples, d)
        d = all_embeddings.shape[1]
        sample_embeddings = sample_embeddings_flat.reshape(n, self.n_samples, d)

        # Step 3 & 4: Compute entropy and consistency per prompt
        semantic_entropies = np.zeros(n)
        self_consistencies = np.zeros(n)

        for i in range(n):
            samples_emb = sample_embeddings[i]  # (n_samples, d)
            orig_emb = orig_embeddings[i]        # (d,)

            # Semantic entropy
            semantic_entropies[i] = self._compute_semantic_entropy(samples_emb)

            # Self-consistency
            self_consistencies[i] = self._compute_self_consistency(
                orig_emb, samples_emb
            )

        # Step 5: Combine
        # Normalize entropy to [0, 1]
        max_entropy = np.log(self.n_samples) if self.n_samples > 1 else 1.0
        normalized_entropy = semantic_entropies / max_entropy
        normalized_entropy = np.clip(normalized_entropy, 0.0, 1.0)

        # Final score: high = good (confident), low = bad (uncertain)
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

        Algorithm:
        1. Compute pairwise cosine similarity matrix (n_samples x n_samples).
        2. Greedy clustering into semantic equivalence classes:
           a. Mark all samples as unassigned.
           b. While unassigned samples remain:
              - Pick the first unassigned sample as a new cluster center.
              - Assign all unassigned samples with sim > threshold to this cluster.
           c. Result: list of cluster sizes.
        3. Compute probabilities: p_k = cluster_size_k / n_samples
        4. Entropy = -sum(p_k * log(p_k))

        Edge cases:
        - n_samples = 1: entropy = 0 (no variation to measure)
        - All samples identical: 1 cluster, entropy = 0
        - All samples different: n_samples clusters, entropy = log(n_samples)
        """
        n = samples_emb.shape[0]
        if n <= 1:
            return 0.0

        # Normalize
        norms = np.linalg.norm(samples_emb, axis=1, keepdims=True) + 1e-10
        normed = samples_emb / norms

        # Pairwise cosine similarity
        sim_matrix = normed @ normed.T  # (n, n)

        # Greedy clustering
        assigned = np.full(n, -1, dtype=int)
        cluster_id = 0
        for i in range(n):
            if assigned[i] >= 0:
                continue
            # Start new cluster with sample i
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
        # Filter out zero probabilities (shouldn't happen but be safe)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))
        return float(entropy)

    def _compute_self_consistency(
        self,
        orig_emb: np.ndarray,
        samples_emb: np.ndarray,
    ) -> float:
        """
        Compute fraction of samples that agree with the original response.

        Algorithm:
        1. Compute cosine similarity between orig_emb and each sample embedding.
        2. Count fraction where sim >= consistency_threshold.

        Edge cases:
        - Zero-norm original: return 0.0 (no basis for comparison)
        """
        orig_norm = np.linalg.norm(orig_emb)
        if orig_norm < 1e-10:
            return 0.0

        orig_normed = orig_emb / orig_norm
        sample_norms = np.linalg.norm(samples_emb, axis=1) + 1e-10
        samples_normed = samples_emb / sample_norms[:, np.newaxis]

        sims = samples_normed @ orig_normed  # (n_samples,)
        agreement = np.mean(sims >= self.consistency_threshold)
        return float(agreement)
```

### 8e. scoring/__init__.py

**File**: `/Users/gabo/Developer/Gabon/faultmap/faultmap/scoring/__init__.py`

```python
from .base import BaseScorer
from .entropy import EntropyScorer
from .precomputed import PrecomputedScorer
from .reference import ReferenceScorer

__all__ = ["BaseScorer", "PrecomputedScorer", "ReferenceScorer", "EntropyScorer"]
```

---

## 9. Module: slicing/

### 9a. slicing/clustering.py

**File**: `/Users/gabo/Developer/Gabon/faultmap/faultmap/slicing/clustering.py`

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
        embeddings: (n, d) array of embeddings.
        method: "hdbscan" or "agglomerative".
        min_cluster_size: Minimum cluster size for HDBSCAN. Also used as a filter
                          for agglomerative (post-hoc removal of tiny clusters).
        n_clusters: Number of clusters for agglomerative. If None, auto-select
                    using silhouette score over range [5, 10, 15, 20, 25, 30].

    Returns:
        labels: (n,) integer array. -1 = noise (HDBSCAN) or removed (agglomerative).

    Algorithm for HDBSCAN:
    1. from sklearn.cluster import HDBSCAN
    2. clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean",
                           cluster_selection_method="eom")
       NOTE: We use euclidean on L2-normalized embeddings, which is monotonically
       related to cosine distance: ||a-b||^2 = 2 - 2*cos(a,b) for unit vectors.
    3. Normalize embeddings to unit length before clustering.
    4. labels = clusterer.fit_predict(normed_embeddings)
    5. If all labels are -1: raise ClusteringError("HDBSCAN found no clusters. Try reducing min_cluster_size.")

    Algorithm for agglomerative:
    1. from sklearn.cluster import AgglomerativeClustering
    2. Normalize embeddings to unit length.
    3. If n_clusters is None:
       a. Try n_clusters in [5, 10, 15, 20, 25, 30] (filtered to < n // 2)
       b. For each k: fit, compute silhouette score
       c. Pick k with best silhouette
       d. If no valid k (e.g., n < 10): use n_clusters = max(2, n // 5)
    4. clusterer = AgglomerativeClustering(n_clusters=best_k, metric="euclidean", linkage="ward")
    5. labels = clusterer.fit_predict(normed_embeddings)
    6. Post-filter: set labels to -1 for clusters smaller than min_cluster_size.

    Edge cases:
    - n < min_cluster_size: raise ClusteringError with helpful message
    - n < 30: warn that results may be unreliable with small datasets
    - All embeddings identical: HDBSCAN will produce 1 cluster or all noise
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

    # L2-normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    normed = embeddings / norms

    if method == "hdbscan":
        return _cluster_hdbscan(normed, min_cluster_size)
    elif method == "agglomerative":
        return _cluster_agglomerative(normed, min_cluster_size, n_clusters)
    else:
        raise ClusteringError(f"Unknown clustering method: {method!r}. Use 'hdbscan' or 'agglomerative'.")


def _cluster_hdbscan(normed: np.ndarray, min_cluster_size: int) -> np.ndarray:
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

    logger.info(f"HDBSCAN found {len(unique_labels)} clusters, {np.sum(labels == -1)} noise points")
    return labels


def _cluster_agglomerative(
    normed: np.ndarray,
    min_cluster_size: int,
    n_clusters: int | None,
) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    n = normed.shape[0]

    if n_clusters is None:
        # Auto-select
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
                    s = silhouette_score(normed, temp_labels, metric="euclidean", sample_size=min(n, 5000))
                    if s > best_score:
                        best_score = s
                        best_k = k
                except Exception:
                    continue
            n_clusters = best_k

    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = clusterer.fit_predict(normed)

    # Post-filter: remove clusters smaller than min_cluster_size
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
            f"All agglomerative clusters were smaller than min_cluster_size={min_cluster_size}. "
            f"Try reducing min_cluster_size."
        )

    logger.info(f"Agglomerative clustering: {len(remaining)} clusters (n_clusters param={n_clusters})")
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
    1. Get indices where labels == cluster_id.
    2. Compute centroid = mean of those embeddings.
    3. Compute cosine similarity of each member to centroid.
    4. Return top_k by similarity.

    Returns:
        (representative_prompts, representative_indices)
    """
    member_mask = labels == cluster_id
    member_indices = np.where(member_mask)[0]
    member_embs = embeddings[member_mask]

    centroid = member_embs.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid) + 1e-10
    centroid = centroid / centroid_norm

    emb_norms = np.linalg.norm(member_embs, axis=1, keepdims=True) + 1e-10
    normed_embs = member_embs / emb_norms

    sims = normed_embs @ centroid  # (m,)
    top_local = np.argsort(sims)[::-1][:top_k]

    top_global_indices = member_indices[top_local]
    top_prompts = [prompts[idx] for idx in top_global_indices]

    return top_prompts, top_global_indices.tolist()
```

### 9b. slicing/statistics.py

**File**: `/Users/gabo/Developer/Gabon/faultmap/faultmap/slicing/statistics.py`

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
    test_used: str          # "chi2" | "fisher"
    adjusted_p_value: float = 1.0  # Set after BH correction


def test_cluster_failure_rate(
    cluster_failures: int,
    cluster_size: int,
    total_failures: int,
    total_size: int,
    cluster_id: int,
) -> ClusterTestResult:
    """
    Test whether a cluster's failure rate is significantly higher than baseline.

    Decision logic for which test to use:
    - If ANY cell in the 2x2 contingency table has expected count < 5: use Fisher exact test
    - Otherwise: use chi-squared test with Yates' continuity correction

    The 2x2 contingency table:
                        Failed    Passed
    In cluster:           a         b        | cluster_size
    Not in cluster:       c         d        | total_size - cluster_size
                        ----      ----
                     total_fail  total_pass   | total_size

    Where:
        a = cluster_failures
        b = cluster_size - cluster_failures
        c = total_failures - cluster_failures
        d = (total_size - cluster_size) - (total_failures - cluster_failures)

    Algorithm for chi-squared (with Yates correction):
    1. Compute expected values: E_ij = (row_total_i * col_total_j) / total_size
    2. If min(E_ij) < 5: switch to Fisher
    3. chi2 = sum((|O_ij - E_ij| - 0.5)^2 / E_ij)  (Yates correction)
    4. p-value from chi2 survival function with df=1
       Implemented manually: p = erfc(sqrt(chi2/2) / sqrt(2)) -- but this is the
       same as the regularized incomplete gamma function. Simpler: use the relation
       chi2 with 1 df: p = 2 * (1 - Phi(sqrt(chi2))) where Phi is normal CDF.
       Even simpler: use scipy-free approximation or import from math.
       DECISION: Use the survival function of chi2(df=1) via the complementary
       error function: p = erfc(sqrt(chi2_stat / 2)) which is exact for df=1.

    Algorithm for Fisher exact test (implemented without scipy):
    1. The exact p-value for a 2x2 table with fixed margins is:
       p = sum of P(X >= a) where X ~ Hypergeometric(total_size, total_failures, cluster_size)
    2. Implemented via log-factorials to avoid overflow.
    3. One-sided (greater): we only care if failure rate is HIGHER than baseline.

    Edge cases:
    - cluster_failures = 0: failure rate is 0, clearly not significantly higher -> p = 1.0
    - cluster_failures = cluster_size: check if this is significant vs baseline
    - cluster_size = total_size: the cluster IS the whole dataset -> p = 1.0
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

    # Compute expected values
    row_totals = [cluster_size, total_size - cluster_size]
    col_totals = [total_failures, total_size - total_failures]
    expected = np.array([
        [row_totals[0] * col_totals[0] / total_size,
         row_totals[0] * col_totals[1] / total_size],
        [row_totals[1] * col_totals[0] / total_size,
         row_totals[1] * col_totals[1] / total_size],
    ])

    if np.min(expected) < 5:
        # Fisher exact test
        p_value = _fisher_exact_one_sided(a, b, c, d)
        test_used = "fisher"
    else:
        # Chi-squared with Yates correction
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
    Chi-squared test with Yates' continuity correction for 2x2 table.

    Formula: chi2 = n * (|ad - bc| - n/2)^2 / ((a+b)(c+d)(a+c)(b+d))

    p-value for df=1: p = erfc(sqrt(chi2/2))
    This uses the relationship: for chi2 with 1 df, the survival function is
    P(X > x) = erfc(sqrt(x/2)) where erfc is the complementary error function.

    We divide by 2 at the end for one-sided test (we only care about higher failure rate).
    """
    from math import erfc, sqrt

    numerator = (abs(a * d - b * c) - n / 2.0) ** 2 * n
    denominator = (a + b) * (c + d) * (a + c) * (b + d)

    if denominator == 0:
        return 1.0

    chi2_stat = numerator / denominator

    # Two-sided p-value from chi2(df=1)
    p_two_sided = erfc(sqrt(chi2_stat / 2.0))

    # One-sided: divide by 2 (we only care about failure rate being HIGHER)
    return p_two_sided / 2.0


def _fisher_exact_one_sided(a: int, b: int, c: int, d: int) -> float:
    """
    Fisher exact test, one-sided (greater), without scipy.

    Uses the hypergeometric distribution. We want P(X >= a) where
    X ~ Hypergeometric(N=a+b+c+d, K=a+c, n=a+b).

    Algorithm:
    1. Compute log-probability of each table with X = a, a+1, ..., min(a+b, a+c)
       using log-factorials.
    2. P(table with x) = C(K,x) * C(N-K, n-x) / C(N, n)
       In log form: log_choose(K,x) + log_choose(N-K, n-x) - log_choose(N, n)
    3. Sum probabilities for x >= a.

    Edge cases:
    - Very large tables: use Stirling or pre-computed log-factorials up to N.
    - a = 0: P(X >= 0) = 1.0
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
    1. Sort results by p_value ascending.
    2. m = number of tests.
    3. For each result at rank i (1-indexed):
       adjusted_p = p_value * m / i
    4. Enforce monotonicity: walk backwards, each adjusted_p = min(adjusted_p, next_adjusted_p).
    5. Clip to [0, 1].
    6. Assign adjusted_p_value to each result.
    7. Return results sorted by adjusted_p_value ascending.

    Edge cases:
    - Empty list: return empty
    - Single test: adjusted_p = p_value (no correction needed with m=1, it simplifies to p*1/1)
    - All p-values = 1.0: all adjusted = 1.0
    """
    if not results:
        return []

    # Sort by p_value ascending
    sorted_results = sorted(results, key=lambda r: r.p_value)
    m = len(sorted_results)

    # Compute raw adjusted p-values
    adjusted = [0.0] * m
    for i, result in enumerate(sorted_results):
        rank = i + 1  # 1-indexed
        adjusted[i] = result.p_value * m / rank

    # Enforce monotonicity (backwards pass)
    for i in range(m - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    # Clip and assign
    for i, result in enumerate(sorted_results):
        result.adjusted_p_value = min(max(adjusted[i], 0.0), 1.0)

    # Return sorted by adjusted p-value
    return sorted(sorted_results, key=lambda r: r.adjusted_p_value)
```

### 9c. slicing/__init__.py

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

## 10. Module: coverage/

### 10a. coverage/detector.py

**File**: `/Users/gabo/Developer/Gabon/faultmap/faultmap/coverage/detector.py`

```python
from __future__ import annotations
import logging

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..exceptions import ClusteringError
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
    Find production prompts that are far from any test prompt.

    Args:
        test_embeddings: (n_test, d) embeddings of test prompts.
        prod_embeddings: (n_prod, d) embeddings of production prompts.
        prod_prompts: Original production prompt strings.
        distance_threshold: Distance above which a prod prompt is "uncovered".
            If None: auto-compute as mean + 1.5 * std of NN distances.
        min_gap_size: Minimum number of uncovered prompts to form a gap cluster.
        clustering_method: "hdbscan" or "agglomerative" for clustering the gaps.

    Returns:
        (gap_labels, nn_distances, distance_threshold)
        gap_labels: (n_prod,) array, -1 for covered prompts, cluster_id for gap members.
        nn_distances: (n_prod,) array of nearest-neighbor distances.
        distance_threshold: The threshold used.

    Algorithm:
    1. L2-normalize both test and prod embeddings.
    2. Fit NearestNeighbors on test_embeddings (k=1, metric="euclidean").
    3. Query with prod_embeddings to get distances.
    4. If distance_threshold is None:
       auto = mean(distances) + 1.5 * std(distances)
       distance_threshold = auto
    5. uncovered_mask = distances > distance_threshold
    6. If sum(uncovered_mask) < min_gap_size:
       - No significant gaps. Return all -1 labels.
    7. Cluster the uncovered prod embeddings using cluster_embeddings().
    8. Map cluster labels back to full prod array (covered prompts get -1).

    Edge cases:
    - No test prompts: raise ConfigurationError
    - No prod prompts: return empty arrays
    - All prod prompts are close: no gaps found (valid result)
    - Very few uncovered points: may fail to cluster -> return all as one gap
    """
    from ..exceptions import ConfigurationError

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
    distances = distances.ravel()  # (n_prod,)

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
        # If clustering fails, treat all uncovered as one gap
        logger.warning("Clustering of uncovered prompts failed. Treating all as one gap.")
        uncovered_cluster_labels = np.zeros(n_uncovered, dtype=int)

    # Map back
    uncovered_indices = np.where(uncovered_mask)[0]
    for local_idx, global_idx in enumerate(uncovered_indices):
        gap_labels[global_idx] = uncovered_cluster_labels[local_idx]

    return gap_labels, distances, distance_threshold
```

### 10b. coverage/__init__.py

```python
from .detector import detect_coverage_gaps

__all__ = ["detect_coverage_gaps"]
```

---

## 11. Module: report.py

**File**: `/Users/gabo/Developer/Gabon/faultmap/faultmap/report.py`

```python
from __future__ import annotations

from .models import AnalysisReport, CoverageReport


def format_analysis_report(report: AnalysisReport) -> str:
    """
    Format an AnalysisReport as a human-readable string.

    Uses rich if available, otherwise plain text.

    Plain text format:
    ═══════════════════════════════════════════
    FAULTMAP ANALYSIS REPORT
    ═══════════════════════════════════════════
    Total prompts:     {N}
    Total failures:    {M} ({rate:.1%})
    Scoring mode:      {mode}
    Clustering:        {method}
    Significance:      alpha={alpha}
    Clusters tested:   {K}
    Significant:       {S}
    ───────────────────────────────────────────
    [For each significant slice, sorted by adjusted p-value:]

    Slice 1: "{name}"
      Description:   {description}
      Size:          {size} prompts
      Failure rate:  {rate:.1%} (vs {baseline:.1%} baseline)
      p-value:       {p:.4f} (adjusted: {adj_p:.4f})
      Test:          {test_used}
      Examples:
        - {prompt_1}
        - {prompt_2}
        - {prompt_3}
    ───────────────────────────────────────────
    ...
    ═══════════════════════════════════════════
    """
    try:
        return _format_analysis_rich(report)
    except ImportError:
        return _format_analysis_plain(report)


def format_coverage_report(report: CoverageReport) -> str:
    """
    Format a CoverageReport. Same rich/plain fallback pattern.

    Plain text format:
    ═══════════════════════════════════════════
    FAULTMAP COVERAGE REPORT
    ═══════════════════════════════════════════
    Test prompts:        {N}
    Production prompts:  {M}
    Coverage score:      {score:.1%}
    Distance threshold:  {thresh:.4f}
    Gaps found:          {G}
    ───────────────────────────────────────────
    [For each gap:]

    Gap 1: "{name}"
      Description:     {description}
      Size:            {size} prompts
      Mean distance:   {dist:.4f}
      Examples:
        - {prompt_1}
        - ...
    ───────────────────────────────────────────
    """
    try:
        return _format_coverage_rich(report)
    except ImportError:
        return _format_coverage_plain(report)


def _format_analysis_plain(report: AnalysisReport) -> str:
    """Plain-text formatting for AnalysisReport."""
    sep = "=" * 50
    thin_sep = "-" * 50
    lines = [
        sep,
        "FAULTMAP ANALYSIS REPORT",
        sep,
        f"Total prompts:     {report.total_prompts}",
        f"Total failures:    {report.total_failures} ({report.baseline_failure_rate:.1%})",
        f"Scoring mode:      {report.scoring_mode}",
        f"Clustering:        {report.clustering_method}",
        f"Embedding model:   {report.embedding_model}",
        f"Significance:      alpha={report.significance_level}",
        f"Clusters tested:   {report.num_clusters_tested}",
        f"Significant:       {report.num_significant}",
    ]

    if not report.slices:
        lines.append(thin_sep)
        lines.append("No statistically significant failure slices found.")
    else:
        for i, s in enumerate(report.slices, 1):
            lines.append(thin_sep)
            lines.append(f'Slice {i}: "{s.name}"')
            lines.append(f"  Description:   {s.description}")
            lines.append(f"  Size:          {s.size} prompts")
            lines.append(f"  Failure rate:  {s.failure_rate:.1%} (vs {s.baseline_failure_rate:.1%} baseline)")
            lines.append(f"  p-value:       {s.p_value:.6f} (adjusted: {s.adjusted_p_value:.6f})")
            lines.append(f"  Test:          {s.test_used}")
            lines.append(f"  Examples:")
            for prompt in s.representative_prompts[:5]:
                truncated = prompt[:120] + ("..." if len(prompt) > 120 else "")
                lines.append(f"    - {truncated}")

    lines.append(sep)
    return "\n".join(lines)


def _format_analysis_rich(report: AnalysisReport) -> str:
    """Rich-formatted table for AnalysisReport. Raises ImportError if rich not installed."""
    from rich.console import Console
    from rich.table import Table
    from io import StringIO

    console = Console(file=StringIO(), force_terminal=True, width=120)

    console.print("[bold]FAULTMAP ANALYSIS REPORT[/bold]", style="cyan")
    console.print(f"Prompts: {report.total_prompts} | "
                  f"Failures: {report.total_failures} ({report.baseline_failure_rate:.1%}) | "
                  f"Mode: {report.scoring_mode} | "
                  f"Clustering: {report.clustering_method}")
    console.print(f"Clusters tested: {report.num_clusters_tested} | "
                  f"Significant: {report.num_significant} | "
                  f"Alpha: {report.significance_level}")

    if report.slices:
        table = Table(show_header=True, header_style="bold")
        table.add_column("#", width=3)
        table.add_column("Name", width=25)
        table.add_column("Size", width=6, justify="right")
        table.add_column("Fail Rate", width=10, justify="right")
        table.add_column("Baseline", width=10, justify="right")
        table.add_column("Adj. p", width=10, justify="right")
        table.add_column("Test", width=6)

        for i, s in enumerate(report.slices, 1):
            table.add_row(
                str(i),
                s.name,
                str(s.size),
                f"{s.failure_rate:.1%}",
                f"{s.baseline_failure_rate:.1%}",
                f"{s.adjusted_p_value:.4f}",
                s.test_used,
            )

        console.print(table)

        for i, s in enumerate(report.slices, 1):
            console.print(f"\n[bold]Slice {i}: {s.name}[/bold]")
            console.print(f"  {s.description}")
            console.print("  Examples:")
            for p in s.representative_prompts[:3]:
                truncated = p[:100] + ("..." if len(p) > 100 else "")
                console.print(f"    - {truncated}", style="dim")
    else:
        console.print("[green]No statistically significant failure slices found.[/green]")

    return console.file.getvalue()


def _format_coverage_plain(report: CoverageReport) -> str:
    """Plain-text formatting for CoverageReport."""
    sep = "=" * 50
    thin_sep = "-" * 50
    lines = [
        sep,
        "FAULTMAP COVERAGE REPORT",
        sep,
        f"Test prompts:        {report.num_test_prompts}",
        f"Production prompts:  {report.num_production_prompts}",
        f"Coverage score:      {report.overall_coverage_score:.1%}",
        f"Distance threshold:  {report.distance_threshold:.4f}",
        f"Embedding model:     {report.embedding_model}",
        f"Gaps found:          {report.num_gaps}",
    ]

    if not report.gaps:
        lines.append(thin_sep)
        lines.append("No significant coverage gaps found.")
    else:
        for i, g in enumerate(report.gaps, 1):
            lines.append(thin_sep)
            lines.append(f'Gap {i}: "{g.name}"')
            lines.append(f"  Description:     {g.description}")
            lines.append(f"  Size:            {g.size} prompts")
            lines.append(f"  Mean distance:   {g.mean_distance:.4f}")
            lines.append(f"  Examples:")
            for prompt in g.representative_prompts[:5]:
                truncated = prompt[:120] + ("..." if len(prompt) > 120 else "")
                lines.append(f"    - {truncated}")

    lines.append(sep)
    return "\n".join(lines)


def _format_coverage_rich(report: CoverageReport) -> str:
    """Rich-formatted table for CoverageReport. Raises ImportError if rich not installed."""
    from rich.console import Console
    from rich.table import Table
    from io import StringIO

    console = Console(file=StringIO(), force_terminal=True, width=120)

    console.print("[bold]FAULTMAP COVERAGE REPORT[/bold]", style="cyan")
    console.print(f"Test prompts: {report.num_test_prompts} | "
                  f"Production prompts: {report.num_production_prompts} | "
                  f"Coverage: {report.overall_coverage_score:.1%}")

    if report.gaps:
        table = Table(show_header=True, header_style="bold")
        table.add_column("#", width=3)
        table.add_column("Name", width=25)
        table.add_column("Size", width=6, justify="right")
        table.add_column("Mean Dist", width=10, justify="right")

        for i, g in enumerate(report.gaps, 1):
            table.add_row(str(i), g.name, str(g.size), f"{g.mean_distance:.4f}")

        console.print(table)
    else:
        console.print("[green]No significant coverage gaps found.[/green]")

    return console.file.getvalue()
```

---

## 12. Module: analyzer.py -- THE ORCHESTRATOR

**File**: `/Users/gabo/Developer/Gabon/faultmap/faultmap/analyzer.py`

```python
from __future__ import annotations
import logging
import warnings
from typing import Optional

import numpy as np

from .embeddings import Embedder, get_embedder
from .exceptions import ConfigurationError
from .labeling import label_clusters
from .llm import AsyncLLMClient
from .models import (
    AnalysisReport,
    CoverageGap,
    CoverageReport,
    FailureSlice,
    ScoringResult,
)
from .report import format_analysis_report, format_coverage_report
from .utils import run_sync, validate_inputs

logger = logging.getLogger(__name__)


class SliceAnalyzer:
    """
    Main entry point for faultmap.

    Constructor params (all stored as instance attributes):
        model: str = "gpt-4o-mini"
            litellm model string for LLM calls (labeling, entropy scoring).
        embedding_model: str = "all-MiniLM-L6-v2"
            Embedding model. Auto-detected as local vs API.
        significance_level: float = 0.05
            Alpha for BH correction.
        min_slice_size: int = 10
            Minimum cluster size for both HDBSCAN and post-filtering.
        failure_threshold: float = 0.5
            Score below this = failure. Used in Mode 1 and Mode 2.
            Mode 3 also uses this to binarize the confidence score.
        n_samples: int = 8
            Number of LLM samples per prompt for Mode 3.
        clustering_method: str = "hdbscan"
            "hdbscan" or "agglomerative".
        max_concurrent_requests: int = 50
            Semaphore limit for async LLM calls.
        temperature: float = 1.0
            Sampling temperature for Mode 3.
        consistency_threshold: float = 0.8
            Cosine similarity threshold for semantic equivalence in Mode 3.

    Internal objects (created lazily or in constructor):
        self._embedder: Embedder (created in __init__ via get_embedder)
        self._llm_client: AsyncLLMClient (created in __init__)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        embedding_model: str = "all-MiniLM-L6-v2",
        significance_level: float = 0.05,
        min_slice_size: int = 10,
        failure_threshold: float = 0.5,
        n_samples: int = 8,
        clustering_method: str = "hdbscan",
        max_concurrent_requests: int = 50,
        temperature: float = 1.0,
        consistency_threshold: float = 0.8,
    ) -> None:
        # Validate config
        if clustering_method not in ("hdbscan", "agglomerative"):
            raise ConfigurationError(
                f"clustering_method must be 'hdbscan' or 'agglomerative', got {clustering_method!r}"
            )
        if not 0 < significance_level < 1:
            raise ConfigurationError("significance_level must be in (0, 1)")
        if not 0 <= failure_threshold <= 1:
            raise ConfigurationError("failure_threshold must be in [0, 1]")
        if n_samples < 2:
            raise ConfigurationError("n_samples must be >= 2 for entropy scoring")

        self.model = model
        self.embedding_model = embedding_model
        self.significance_level = significance_level
        self.min_slice_size = min_slice_size
        self.failure_threshold = failure_threshold
        self.n_samples = n_samples
        self.clustering_method = clustering_method
        self.max_concurrent_requests = max_concurrent_requests
        self.temperature = temperature
        self.consistency_threshold = consistency_threshold

        # Create internal objects
        self._embedder: Embedder = get_embedder(embedding_model)
        self._llm_client = AsyncLLMClient(
            model=model,
            max_concurrent_requests=max_concurrent_requests,
        )

    def analyze(
        self,
        prompts: list[str],
        responses: list[str],
        scores: Optional[list[float]] = None,
        references: Optional[list[str]] = None,
    ) -> AnalysisReport:
        """
        Synchronous entry point. Discovers failure slices.

        Mode detection:
        - scores is not None -> Mode 1 (precomputed)
        - references is not None -> Mode 2 (reference-based)
        - Both None -> Mode 3 (entropy)
        - Both provided -> scores wins with warning

        Returns:
            AnalysisReport

        Delegates to _analyze_async() via run_sync().
        """
        return run_sync(self._analyze_async(prompts, responses, scores, references))

    async def _analyze_async(
        self,
        prompts: list[str],
        responses: list[str],
        scores: Optional[list[float]],
        references: Optional[list[str]],
    ) -> AnalysisReport:
        """
        Async orchestration pipeline.

        Steps:
        1. VALIDATE inputs
        2. DETECT scoring mode and warn if ambiguous
        3. SCORE: Run appropriate scorer to get scores in [0, 1]
        4. BINARIZE: failures = [i for i, s in enumerate(scores) if s < failure_threshold]
        5. EMBED: Embed prompts (NOT responses -- we cluster by input similarity)
        6. CLUSTER: cluster_embeddings(prompt_embeddings, method, min_cluster_size)
        7. TEST: For each cluster, run statistical test (chi2/Fisher)
        8. CORRECT: Apply BH correction
        9. FILTER: Keep clusters with adjusted_p_value < significance_level
        10. NAME: Use LLM to name significant clusters
        11. ASSEMBLE: Build FailureSlice objects and AnalysisReport
        12. LOG: Print formatted report summary to logger
        """
        from .scoring import PrecomputedScorer, ReferenceScorer, EntropyScorer
        from .slicing import (
            cluster_embeddings,
            get_representative_prompts,
            test_cluster_failure_rate,
            benjamini_hochberg,
        )

        # Step 1: Validate
        validate_inputs(prompts, responses, scores, references)

        # Step 2: Mode detection
        if scores is not None and references is not None:
            warnings.warn(
                "Both scores and references provided. Using scores (Mode 1).",
                UserWarning,
                stacklevel=3,
            )
            references = None

        if scores is not None:
            scoring_mode = "precomputed"
            scorer = PrecomputedScorer(scores)
        elif references is not None:
            scoring_mode = "reference"
            scorer = ReferenceScorer(self._embedder, references)
        else:
            scoring_mode = "entropy"
            scorer = EntropyScorer(
                client=self._llm_client,
                embedder=self._embedder,
                n_samples=self.n_samples,
                temperature=self.temperature,
                consistency_threshold=self.consistency_threshold,
            )

        # Step 3: Score
        logger.info(f"Scoring mode: {scoring_mode}")
        scoring_result: ScoringResult = await scorer.score(prompts, responses)

        # Step 4: Binarize
        score_array = np.array(scoring_result.scores)
        failures = score_array < self.failure_threshold
        total_failures = int(np.sum(failures))
        total_prompts = len(prompts)
        baseline_failure_rate = total_failures / total_prompts if total_prompts > 0 else 0.0

        logger.info(
            f"Failures: {total_failures}/{total_prompts} ({baseline_failure_rate:.1%}) "
            f"at threshold={self.failure_threshold}"
        )

        if total_failures == 0:
            logger.info("No failures detected. Returning empty report.")
            return AnalysisReport(
                slices=[],
                total_prompts=total_prompts,
                total_failures=0,
                baseline_failure_rate=0.0,
                significance_level=self.significance_level,
                failure_threshold=self.failure_threshold,
                scoring_mode=scoring_mode,
                num_clusters_tested=0,
                num_significant=0,
                clustering_method=self.clustering_method,
                embedding_model=self.embedding_model,
            )

        # Step 5: Embed prompts
        logger.info("Embedding prompts...")
        prompt_embeddings = self._embedder.embed(prompts)

        # Step 6: Cluster
        logger.info(f"Clustering ({self.clustering_method})...")
        labels = cluster_embeddings(
            prompt_embeddings,
            method=self.clustering_method,
            min_cluster_size=self.min_slice_size,
        )

        unique_labels = sorted(set(labels))
        if -1 in unique_labels:
            unique_labels.remove(-1)

        # Step 7: Statistical testing
        logger.info(f"Testing {len(unique_labels)} clusters...")
        test_results = []
        for cid in unique_labels:
            cluster_mask = labels == cid
            cluster_size = int(np.sum(cluster_mask))
            cluster_failures = int(np.sum(failures[cluster_mask]))
            result = test_cluster_failure_rate(
                cluster_failures=cluster_failures,
                cluster_size=cluster_size,
                total_failures=total_failures,
                total_size=total_prompts,
                cluster_id=cid,
            )
            test_results.append(result)

        # Step 8: BH correction
        corrected = benjamini_hochberg(test_results, alpha=self.significance_level)

        # Step 9: Filter significant
        significant = [r for r in corrected if r.adjusted_p_value < self.significance_level]
        logger.info(
            f"{len(significant)}/{len(corrected)} clusters are significant "
            f"at alpha={self.significance_level}"
        )

        num_clusters_tested = len(corrected)

        if not significant:
            return AnalysisReport(
                slices=[],
                total_prompts=total_prompts,
                total_failures=total_failures,
                baseline_failure_rate=baseline_failure_rate,
                significance_level=self.significance_level,
                failure_threshold=self.failure_threshold,
                scoring_mode=scoring_mode,
                num_clusters_tested=num_clusters_tested,
                num_significant=0,
                clustering_method=self.clustering_method,
                embedding_model=self.embedding_model,
            )

        # Step 10: Name significant clusters
        clusters_texts = []
        clusters_indices = []
        for r in significant:
            rep_prompts, rep_indices = get_representative_prompts(
                prompt_embeddings, labels, r.cluster_id, prompts, top_k=10
            )
            clusters_texts.append(rep_prompts)
            clusters_indices.append(rep_indices)

        logger.info(f"Naming {len(significant)} significant clusters...")
        cluster_labels = await label_clusters(
            self._llm_client, clusters_texts, context="failure slice"
        )

        # Step 11: Assemble
        slices: list[FailureSlice] = []
        for r, label, rep_texts, rep_indices in zip(
            significant, cluster_labels, clusters_texts, clusters_indices
        ):
            # Get top-5 for the report (we used top-10 for naming)
            slice_obj = FailureSlice(
                name=label.name,
                description=label.description,
                size=r.size,
                failure_rate=r.failure_rate,
                baseline_failure_rate=baseline_failure_rate,
                p_value=r.p_value,
                adjusted_p_value=r.adjusted_p_value,
                test_used=r.test_used,
                representative_prompts=rep_texts[:5],
                prompt_indices=[int(idx) for idx in rep_indices[:5]],
                cluster_id=r.cluster_id,
            )
            slices.append(slice_obj)

        report = AnalysisReport(
            slices=slices,
            total_prompts=total_prompts,
            total_failures=total_failures,
            baseline_failure_rate=baseline_failure_rate,
            significance_level=self.significance_level,
            failure_threshold=self.failure_threshold,
            scoring_mode=scoring_mode,
            num_clusters_tested=num_clusters_tested,
            num_significant=len(slices),
            clustering_method=self.clustering_method,
            embedding_model=self.embedding_model,
            metadata={"scoring_metadata": scoring_result.metadata},
        )

        logger.info(report.summary())
        return report

    def audit_coverage(
        self,
        test_prompts: list[str],
        production_prompts: list[str],
        distance_threshold: float | None = None,
        min_gap_size: int = 5,
    ) -> CoverageReport:
        """
        Synchronous entry point for coverage auditing.

        Delegates to _audit_coverage_async() via run_sync().
        """
        return run_sync(
            self._audit_coverage_async(
                test_prompts, production_prompts, distance_threshold, min_gap_size
            )
        )

    async def _audit_coverage_async(
        self,
        test_prompts: list[str],
        production_prompts: list[str],
        distance_threshold: float | None,
        min_gap_size: int,
    ) -> CoverageReport:
        """
        Async coverage auditing pipeline.

        Steps:
        1. VALIDATE: Both lists non-empty.
        2. EMBED: Embed test_prompts and production_prompts.
        3. DETECT GAPS: detect_coverage_gaps() -> gap_labels, distances, threshold
        4. For each gap cluster:
           a. Get representative prompts
           b. Compute mean distance
        5. NAME: Use LLM to name gaps.
        6. ASSEMBLE: Build CoverageGap objects and CoverageReport.
        """
        from .coverage import detect_coverage_gaps
        from .slicing.clustering import get_representative_prompts

        if not test_prompts:
            raise ConfigurationError("test_prompts must be non-empty")
        if not production_prompts:
            raise ConfigurationError("production_prompts must be non-empty")

        # Step 2: Embed
        logger.info("Embedding test prompts...")
        test_embeddings = self._embedder.embed(test_prompts)
        logger.info("Embedding production prompts...")
        prod_embeddings = self._embedder.embed(production_prompts)

        # Step 3: Detect gaps
        gap_labels, nn_distances, used_threshold = detect_coverage_gaps(
            test_embeddings=test_embeddings,
            prod_embeddings=prod_embeddings,
            prod_prompts=production_prompts,
            distance_threshold=distance_threshold,
            min_gap_size=min_gap_size,
            clustering_method=self.clustering_method,
        )

        # Find unique gap cluster IDs (excluding -1 = covered)
        unique_gaps = sorted(set(gap_labels))
        if -1 in unique_gaps:
            unique_gaps.remove(-1)

        if not unique_gaps:
            uncovered_count = int(np.sum(gap_labels != -1))
            total = len(production_prompts)
            return CoverageReport(
                gaps=[],
                num_test_prompts=len(test_prompts),
                num_production_prompts=total,
                num_gaps=0,
                overall_coverage_score=1.0 - (uncovered_count / total if total > 0 else 0.0),
                distance_threshold=used_threshold,
                embedding_model=self.embedding_model,
            )

        # Step 4: Representative prompts + mean distances per gap
        clusters_texts = []
        gaps_meta = []  # (cluster_id, size, mean_distance, rep_prompts, rep_indices)
        for cid in unique_gaps:
            mask = gap_labels == cid
            size = int(np.sum(mask))
            mean_dist = float(np.mean(nn_distances[mask]))

            rep_prompts, rep_indices = get_representative_prompts(
                prod_embeddings, gap_labels, cid, production_prompts, top_k=10
            )
            clusters_texts.append(rep_prompts)
            gaps_meta.append((cid, size, mean_dist, rep_prompts[:5], rep_indices[:5]))

        # Step 5: Name gaps
        logger.info(f"Naming {len(unique_gaps)} coverage gaps...")
        cluster_labels = await label_clusters(
            self._llm_client, clusters_texts, context="coverage gap"
        )

        # Step 6: Assemble
        total_uncovered = int(np.sum(gap_labels >= 0))
        total = len(production_prompts)

        gaps: list[CoverageGap] = []
        for label, (cid, size, mean_dist, rep_prompts, rep_indices) in zip(
            cluster_labels, gaps_meta
        ):
            gap = CoverageGap(
                name=label.name,
                description=label.description,
                size=size,
                mean_distance=mean_dist,
                representative_prompts=rep_prompts,
                prompt_indices=[int(idx) for idx in rep_indices],
                cluster_id=cid,
            )
            gaps.append(gap)

        # Sort by mean_distance descending
        gaps.sort(key=lambda g: g.mean_distance, reverse=True)

        report = CoverageReport(
            gaps=gaps,
            num_test_prompts=len(test_prompts),
            num_production_prompts=total,
            num_gaps=len(gaps),
            overall_coverage_score=1.0 - (total_uncovered / total if total > 0 else 0.0),
            distance_threshold=used_threshold,
            embedding_model=self.embedding_model,
        )

        logger.info(report.summary())
        return report

    def print_report(self, report: AnalysisReport | CoverageReport) -> None:
        """Print a formatted report to stdout."""
        if isinstance(report, AnalysisReport):
            print(format_analysis_report(report))
        elif isinstance(report, CoverageReport):
            print(format_coverage_report(report))
        else:
            raise TypeError(f"Expected AnalysisReport or CoverageReport, got {type(report)}")
```

---

## 13. Module: __init__.py

**File**: `/Users/gabo/Developer/Gabon/faultmap/faultmap/__init__.py`

```python
"""
faultmap: Automatically discover where and why your LLM is failing.
"""

from .analyzer import SliceAnalyzer
from .models import (
    AnalysisReport,
    CoverageGap,
    CoverageReport,
    FailureSlice,
    ScoringResult,
)
from .exceptions import (
    FaultmapError,
    EmbeddingError,
    ScoringError,
    LLMError,
    ClusteringError,
    ConfigurationError,
)

__version__ = "0.1.0"

__all__ = [
    "SliceAnalyzer",
    "AnalysisReport",
    "FailureSlice",
    "CoverageReport",
    "CoverageGap",
    "ScoringResult",
    "FaultmapError",
    "EmbeddingError",
    "ScoringError",
    "LLMError",
    "ClusteringError",
    "ConfigurationError",
    "__version__",
]
```

---

## 14. Test Strategy

### Directory structure:
```
tests/
├── conftest.py              # Shared fixtures
├── test_utils.py            # Test cosine similarity, validation, etc.
├── test_embeddings.py       # Test embedders
├── test_llm.py              # Test async LLM client
├── test_labeling.py         # Test cluster naming
├── test_scoring/
│   ├── test_precomputed.py
│   ├── test_reference.py
│   └── test_entropy.py
├── test_slicing/
│   ├── test_clustering.py
│   └── test_statistics.py
├── test_coverage/
│   └── test_detector.py
├── test_report.py
└── test_analyzer.py         # Integration tests
```

### conftest.py shared fixtures:

```python
# Key fixtures:

@pytest.fixture
def sample_embeddings():
    """Deterministic synthetic embeddings: 100 points in 3 clusters."""
    rng = np.random.default_rng(42)
    centers = np.array([[1, 0, 0, ...], [0, 1, 0, ...], [0, 0, 1, ...]])  # 3 cluster centers in 384-d
    # Generate 100 points: 40 near center 0, 35 near center 1, 25 near center 2
    # Add Gaussian noise with small sigma
    ...

@pytest.fixture
def mock_embedder():
    """Embedder that returns deterministic embeddings based on text hash."""
    class MockEmbedder(Embedder):
        def embed(self, texts): ...  # hash-based deterministic
        @property
        def dimension(self): return 64
    return MockEmbedder()

@pytest.fixture
def mock_llm_client():
    """AsyncLLMClient mock that returns canned responses."""
    ...
```

### Per-module test strategy:

**test_utils.py**:
- Test `cosine_similarity_matrix` with known vectors (orthogonal, parallel, antiparallel).
- Test `cosine_similarity_pairs` matches element-wise.
- Test `validate_inputs` with all error cases: mismatched lengths, out-of-range scores, empty inputs.
- Test `run_sync` works in normal context (no running loop).

**test_embeddings.py**:
- `LocalEmbedder`: Mock `sentence_transformers` import. Test that it raises `EmbeddingError` with install instructions when not installed. Test embed with mock model.
- `APIEmbedder`: Mock `litellm.embedding`. Test batching logic (e.g., 300 texts with batch_size=128 -> 3 API calls). Test empty input.
- `get_embedder`: Test routing for known local names and API names.

**test_llm.py**:
- Mock `litellm.acompletion` with controlled responses and failures.
- Test retry logic: fail twice, succeed on third.
- Test semaphore: launch 100 concurrent calls with max_concurrent=5, verify at most 5 run simultaneously.
- Test `complete_batch` returns results in correct order.

**test_labeling.py**:
- Mock LLM client to return "Name: X\nDescription: Y".
- Test `_parse_label_response` with well-formed and malformed responses.
- Test `label_clusters` calls label_cluster for each cluster.

**test_scoring/test_precomputed.py**:
- Trivial: passthrough scores, verify mode="precomputed".

**test_scoring/test_reference.py**:
- Use mock embedder with known vectors.
- Verify score = (cosine_sim + 1) / 2.
- Test with identical texts (score ~1.0) and orthogonal texts (score ~0.5).

**test_scoring/test_entropy.py** (most important):
- Mock LLM client to return deterministic "sampled" responses.
- Mock embedder to return pre-defined embeddings.
- **Test case 1**: All samples identical -> entropy=0, self_consistency=1.0, score near 1.0.
- **Test case 2**: All samples different (orthogonal) -> entropy=log(n), self_consistency~0.0, score near 0.0.
- **Test case 3**: Mixed (2 clusters of 4) -> entropy = log(2) / log(8), moderate score.
- Test `_compute_semantic_entropy` directly with synthetic embeddings.
- Test `_compute_self_consistency` directly.

**test_slicing/test_clustering.py**:
- Create synthetic data with 3 known clusters (well-separated Gaussians in high-d space).
- Test HDBSCAN finds ~3 clusters.
- Test agglomerative with auto n_clusters.
- Test `get_representative_prompts` returns points near centroid.
- Test edge cases: too few points, all identical, single cluster.

**test_slicing/test_statistics.py** (critical):
- **Test chi2**: Known 2x2 table with analytically computed expected p-value. E.g., cluster of 50 with 40 failures out of 200 total with 80 failures. Verify p-value is close to scipy.stats.chi2_contingency (if available) or manually computed.
- **Test Fisher**: Small table (cluster of 8 with 7 failures out of 30 with 10 failures). Verify against known hypergeometric calculation.
- **Test decision logic**: Expected count < 5 -> Fisher, >= 5 -> chi2.
- **Test BH correction**: Manual example:
  - 5 p-values: [0.001, 0.01, 0.03, 0.04, 0.06]
  - m=5
  - Adjusted: [0.005, 0.025, 0.05, 0.05, 0.06]
  - At alpha=0.05: first 3 significant.
- **Test BH edge cases**: empty list, single test, all p=1.0.

**test_coverage/test_detector.py**:
- Create test embeddings in one region, prod embeddings in two regions (one overlapping, one distant).
- Verify distant cluster is detected as gap.
- Test auto-threshold computation.
- Test edge case: all prod prompts covered.

**test_report.py**:
- Test plain text formatting with a known AnalysisReport. Check key strings present.
- Test with empty slices (no significant results).
- Test rich formatting with rich installed (or mock).

**test_analyzer.py** (integration):
- Mock both embedder and LLM client.
- Full pipeline test for Mode 1: provide scores, verify clustering + testing + naming flow.
- Full pipeline test for Mode 3: mock LLM sampling responses, verify entropy computation.
- Test mode detection: scores takes priority over references.
- Test audit_coverage end-to-end.

---

## 15. Implementation Schedule (1-Week MVP)

### Day 1 (Monday): Foundation
- **Files**: `pyproject.toml`, `faultmap/__init__.py`, `exceptions.py`, `models.py`, `utils.py`
- **Tests**: `test_utils.py`
- **Milestone**: Package is installable, models are defined, utility functions work.

### Day 2 (Tuesday): LLM + Embeddings
- **Files**: `llm.py`, `embeddings.py`
- **Tests**: `test_llm.py`, `test_embeddings.py`
- **Milestone**: Can make LLM calls and compute embeddings (both local and API).

### Day 3 (Wednesday): Scoring Pipeline
- **Files**: `scoring/base.py`, `scoring/__init__.py`, `scoring/precomputed.py`, `scoring/reference.py`, `scoring/entropy.py`
- **Tests**: `test_scoring/` (all three)
- **Milestone**: All three scoring modes work independently.

### Day 4 (Thursday): Slicing Pipeline
- **Files**: `slicing/clustering.py`, `slicing/statistics.py`, `slicing/__init__.py`, `labeling.py`
- **Tests**: `test_slicing/`, `test_labeling.py`
- **Milestone**: Can cluster embeddings, run statistical tests, apply BH correction, name clusters.

### Day 5 (Friday): Coverage + Reports + Orchestrator
- **Files**: `coverage/detector.py`, `coverage/__init__.py`, `report.py`, `analyzer.py`
- **Tests**: `test_coverage/`, `test_report.py`, `test_analyzer.py`
- **Milestone**: Full `analyzer.analyze()` and `analyzer.audit_coverage()` work end-to-end.

### Day 6 (Saturday): Integration Testing + Polish
- Manual end-to-end test with real LLM (gpt-4o-mini) and real embeddings.
- Fix any issues found during integration.
- Add logging throughout.
- Write README with usage examples.
- Verify `pip install .` and `pip install .[local]` work.

### Day 7 (Sunday): Buffer + Documentation
- Handle any overflow from previous days.
- Add docstrings to all public API.
- Create a simple example script in `examples/`.
- Final review pass.

---

## Appendix: Key Design Decisions Summarized

1. **Embed prompts, not responses** for failure slice clustering. We want to find "what kinds of questions fail", not "what kinds of answers are bad". Responses vary too much.

2. **One-sided tests only**. We only care if a cluster fails MORE than baseline. A cluster that fails less is not interesting for fault-finding.

3. **Fisher vs chi2 decision**: Automatic based on expected cell counts. Fisher is exact but slow for large tables; chi2 is approximate but fast. The expected-count-5 threshold is the standard textbook rule.

4. **BH over Bonferroni**: BH (Benjamini-Hochberg) controls false discovery rate, not family-wise error rate. With potentially dozens of clusters, Bonferroni would be too conservative. BH is the standard in genomics and similar fields where you expect some real signal.

5. **Entropy scorer weighting**: 50/50 split between normalized entropy and self-consistency. This is a reasonable default. Making it configurable adds complexity without clear benefit for MVP.

6. **cosine similarity mapped to [0,1]**: The `(sim + 1) / 2` mapping in Mode 2 ensures scores are always in [0, 1]. Pure cosine_sim can be negative for dissimilar texts.

7. **HDBSCAN from sklearn**: Using `sklearn.cluster.HDBSCAN` (added in sklearn 1.3) avoids the standalone `hdbscan` package which has problematic C extension dependencies.

8. **L2 normalization before clustering**: On L2-normalized vectors, euclidean distance is monotonically related to cosine distance: `||a-b||^2 = 2 - 2*cos(a,b)`. This lets us use euclidean metric (required by Ward linkage) while effectively clustering by cosine similarity.

9. **`erfc` for chi2 p-value**: For chi-squared with 1 degree of freedom, the survival function is exactly `erfc(sqrt(x/2))`. This avoids needing scipy's `chi2.sf()`.

10. **Greedy clustering for semantic entropy**: Rather than running a full clustering algorithm on N=8 samples, a greedy single-pass algorithm suffices. It is O(n^2) but n is tiny (8 by default).
