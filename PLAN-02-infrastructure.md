# Phase 2 — Infrastructure (Day 2)

**Goal**: LLM calling and embedding computation — the two I/O layers everything depends on.

**Files to create**:
- `faultmap/llm.py`
- `faultmap/embeddings.py`
- `faultmap/labeling.py`
- `tests/test_llm.py`
- `tests/test_embeddings.py`
- `tests/test_labeling.py`

**Milestone**: Can make async rate-limited LLM calls and compute embeddings via both local and API backends.

---

## 1. `faultmap/llm.py` — Async LLM Client

Thin async wrapper around `litellm.acompletion` with semaphore-based rate limiting
and exponential backoff retries.

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

    Used by:
    - scoring/entropy.py (sampling N responses per prompt)
    - labeling.py (naming clusters)
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

        Algorithm:
        1. Acquire semaphore
        2. For attempt in range(max_retries):
            a. Call litellm.acompletion(model, messages, temperature, max_tokens, timeout)
            b. On success: return response.choices[0].message.content.strip()
            c. On failure: exponential backoff = 2^attempt seconds, log warning
        3. Raise LLMError with last exception

        Edge cases:
        - LLM returns None content → raise LLMError
        - Rate limit (429) → handled by backoff
        - Timeout → handled by backoff
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
                        f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): "
                        f"{e}. Retrying in {wait}s."
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
        Batch completion with tqdm progress bar.

        Algorithm:
        1. Create async tasks for each message set
        2. Gather with tqdm_asyncio.gather for progress
        3. Return results in original order

        Edge cases:
        - Empty list → return []
        - Single failure in batch → entire gather raises (caller handles)
        """
        if not messages_list:
            return []

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

**Design notes**:
- Semaphore is per-instance. Multiple `AsyncLLMClient` instances have independent limits.
- `timeout` is per-call, not total batch timeout.
- `litellm` is imported inside methods (not at module level) to keep import fast.

---

## 2. `faultmap/embeddings.py` — Dual Embedding Backend

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
        """Embed texts. Returns shape (len(texts), embedding_dim)."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...


class LocalEmbedder(Embedder):
    """
    Embedding via sentence-transformers (local, no API calls).

    Lazy model loading: model is downloaded/loaded on first embed() call.

    Edge cases:
    - sentence-transformers not installed → EmbeddingError with install instructions
    - Empty list → return np.empty((0, dim))
    """

    # Known local model name prefixes for auto-detection in get_embedder()
    LOCAL_MODEL_PREFIXES = (
        "all-MiniLM", "all-mpnet", "paraphrase-", "multi-qa-", "msmarco-",
        "sentence-t5", "e5-", "bge-", "gte-", "nomic-",
        "BAAI/", "intfloat/", "thenlper/",
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

    Edge cases:
    - Empty list with unknown dimension → probe with a dummy call
    - Rate limiting → litellm handles retries internally
    - API error → wrap in EmbeddingError
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
            probe = litellm.embedding(model=self.model_name, input=["dimension probe"])
            self._dimension = len(probe.data[0]["embedding"])
            return np.empty((0, self._dimension), dtype=np.float32)

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                response = litellm.embedding(model=self.model_name, input=batch)
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
    Factory: determine if model_name is local or API-based.

    Decision logic:
    1. Starts with any LocalEmbedder.LOCAL_MODEL_PREFIXES → LocalEmbedder
    2. Contains "/" and org is a known HuggingFace org → LocalEmbedder
    3. Otherwise → APIEmbedder

    Users can always instantiate LocalEmbedder or APIEmbedder directly
    if auto-detection fails.
    """
    name_lower = model_name.lower()

    for prefix in LocalEmbedder.LOCAL_MODEL_PREFIXES:
        if name_lower.startswith(prefix.lower()):
            return LocalEmbedder(model_name)

    if "/" in model_name:
        org = model_name.split("/")[0].lower()
        local_orgs = {
            "sentence-transformers", "baai", "intfloat", "thenlper", "nomic-ai",
        }
        if org in local_orgs:
            return LocalEmbedder(model_name)

    return APIEmbedder(model_name)
```

---

## 3. `faultmap/labeling.py` — LLM Cluster Naming

Shared module used by both failure slice discovery (slicing layer) and coverage
gap detection (coverage layer). Given representative texts from a cluster, asks
the LLM for a concise name and description.

```python
from __future__ import annotations
import asyncio
from dataclasses import dataclass

from .llm import AsyncLLMClient


@dataclass(frozen=True)
class ClusterLabel:
    name: str          # 2-5 word name, e.g. "Date Formatting Queries"
    description: str   # One sentence description


MAX_EXAMPLES_FOR_NAMING = 15
MAX_CHARS_PER_EXAMPLE = 300


async def label_cluster(
    client: AsyncLLMClient,
    representative_texts: list[str],
    context: str = "failure slice",
) -> ClusterLabel:
    """
    Ask the LLM to name a single cluster.

    Algorithm:
    1. Truncate texts to MAX_EXAMPLES_FOR_NAMING, each to MAX_CHARS_PER_EXAMPLE
    2. System prompt: explain task, request "Name: ...\nDescription: ..." format
    3. Call client.complete() with temperature=0
    4. Parse response; fallback: first line = name, rest = description

    Edge cases:
    - LLM doesn't follow format → graceful fallback parsing
    - Very short texts → still works, just less context for naming
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
        stripped = line.strip()
        if stripped.lower().startswith("name:"):
            name = stripped[len("name:"):].strip()
        elif stripped.lower().startswith("description:"):
            description = stripped[len("description:"):].strip()

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
    Label multiple clusters concurrently via asyncio.gather.
    Returns labels in same order as clusters_texts.
    """
    tasks = [
        label_cluster(client, texts, context=context)
        for texts in clusters_texts
    ]
    return list(await asyncio.gather(*tasks))
```

---

## 4. Day 2 Tests

### `tests/test_llm.py`

```python
import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from faultmap.llm import AsyncLLMClient
from faultmap.exceptions import LLMError


@pytest.fixture
def client():
    return AsyncLLMClient(model="gpt-4o-mini", max_concurrent_requests=5, max_retries=3)


class TestAsyncLLMClient:
    @pytest.mark.asyncio
    async def test_complete_success(self, client):
        """Mock litellm.acompletion, verify we get the response content."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello world"

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await client.complete([{"role": "user", "content": "Hi"}])
            assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_complete_retries_on_failure(self, client):
        """Fail twice, succeed on third attempt."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Success"

        mock_acompletion = AsyncMock(
            side_effect=[Exception("fail 1"), Exception("fail 2"), mock_response]
        )

        with patch("litellm.acompletion", mock_acompletion):
            result = await client.complete([{"role": "user", "content": "Hi"}])
            assert result == "Success"
            assert mock_acompletion.call_count == 3

    @pytest.mark.asyncio
    async def test_complete_raises_after_max_retries(self, client):
        """All retries fail → LLMError."""
        mock_acompletion = AsyncMock(side_effect=Exception("always fails"))

        with patch("litellm.acompletion", mock_acompletion):
            with pytest.raises(LLMError, match="failed after 3 retries"):
                await client.complete([{"role": "user", "content": "Hi"}])

    @pytest.mark.asyncio
    async def test_complete_batch_preserves_order(self, client):
        """Batch results come back in same order as input."""
        call_count = 0

        async def mock_acompletion(**kwargs):
            nonlocal call_count
            call_count += 1
            idx = call_count
            await asyncio.sleep(0.01 * (3 - idx))  # reverse delay
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = f"response-{idx}"
            return mock_resp

        messages_list = [
            [{"role": "user", "content": f"prompt-{i}"}] for i in range(3)
        ]

        with patch("litellm.acompletion", side_effect=mock_acompletion):
            results = await client.complete_batch(
                messages_list, show_progress=False
            )
            assert results == ["response-1", "response-2", "response-3"]

    @pytest.mark.asyncio
    async def test_complete_batch_empty(self, client):
        result = await client.complete_batch([], show_progress=False)
        assert result == []
```

### `tests/test_embeddings.py`

```python
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from faultmap.embeddings import LocalEmbedder, APIEmbedder, get_embedder
from faultmap.exceptions import EmbeddingError


class TestLocalEmbedder:
    def test_raises_when_not_installed(self):
        """Should raise EmbeddingError with install instructions."""
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            embedder = LocalEmbedder("all-MiniLM-L6-v2")
            with pytest.raises(EmbeddingError, match="pip install faultmap"):
                embedder.embed(["test"])

    def test_empty_input(self):
        """Empty list returns (0, dim) array."""
        embedder = LocalEmbedder.__new__(LocalEmbedder)
        embedder._dimension = 64
        embedder._model = MagicMock()
        result = embedder.embed([])
        assert result.shape == (0, 64)


class TestAPIEmbedder:
    def test_batching(self):
        """300 texts with batch_size=128 should make 3 API calls."""
        embedder = APIEmbedder("text-embedding-3-small", batch_size=128)
        texts = [f"text-{i}" for i in range(300)]
        dim = 16

        def mock_embedding(**kwargs):
            batch = kwargs["input"]
            mock_resp = MagicMock()
            mock_resp.data = [
                {"embedding": list(np.random.randn(dim)), "index": i}
                for i in range(len(batch))
            ]
            return mock_resp

        with patch("litellm.embedding", side_effect=mock_embedding) as mock:
            result = embedder.embed(texts)
            assert result.shape == (300, dim)
            assert mock.call_count == 3  # ceil(300/128) = 3

    def test_empty_probes_dimension(self):
        """Empty list should probe API for dimension."""
        embedder = APIEmbedder("text-embedding-3-small")

        mock_resp = MagicMock()
        mock_resp.data = [{"embedding": [0.0] * 1536}]

        with patch("litellm.embedding", return_value=mock_resp):
            result = embedder.embed([])
            assert result.shape == (0, 1536)


class TestGetEmbedder:
    def test_local_model_names(self):
        assert isinstance(get_embedder("all-MiniLM-L6-v2"), LocalEmbedder)
        assert isinstance(get_embedder("all-mpnet-base-v2"), LocalEmbedder)
        assert isinstance(get_embedder("bge-small-en-v1.5"), LocalEmbedder)
        assert isinstance(get_embedder("BAAI/bge-large-en"), LocalEmbedder)

    def test_api_model_names(self):
        assert isinstance(get_embedder("text-embedding-3-small"), APIEmbedder)
        assert isinstance(get_embedder("text-embedding-ada-002"), APIEmbedder)
```

### `tests/test_labeling.py`

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from faultmap.labeling import (
    label_cluster, label_clusters, _parse_label_response, ClusterLabel,
)


class TestParseLabelResponse:
    def test_well_formed(self):
        response = "Name: Legal Questions\nDescription: Questions about legal compliance."
        label = _parse_label_response(response)
        assert label.name == "Legal Questions"
        assert label.description == "Questions about legal compliance."

    def test_malformed_fallback(self):
        response = "These are billing inquiries\nabout payment disputes"
        label = _parse_label_response(response)
        assert label.name == "These are billing inquiries"
        assert "payment disputes" in label.description

    def test_extra_whitespace(self):
        response = "  Name:   Billing Issues  \n  Description:   About billing.  "
        label = _parse_label_response(response)
        assert label.name == "Billing Issues"
        assert label.description == "About billing."


class TestLabelCluster:
    @pytest.mark.asyncio
    async def test_calls_llm_and_parses(self):
        mock_client = AsyncMock()
        mock_client.complete.return_value = (
            "Name: Setup Questions\nDescription: Questions about initial setup."
        )

        label = await label_cluster(
            mock_client,
            ["How do I set up X?", "Setup guide for Y"],
            context="failure slice",
        )

        assert label.name == "Setup Questions"
        mock_client.complete.assert_called_once()


class TestLabelClusters:
    @pytest.mark.asyncio
    async def test_labels_multiple_concurrently(self):
        mock_client = AsyncMock()
        mock_client.complete.side_effect = [
            "Name: Group A\nDescription: First group.",
            "Name: Group B\nDescription: Second group.",
        ]

        labels = await label_clusters(
            mock_client,
            [["text1", "text2"], ["text3", "text4"]],
            context="failure slice",
        )

        assert len(labels) == 2
        assert labels[0].name == "Group A"
        assert labels[1].name == "Group B"
```

---

## Verification

After completing Day 2:
```bash
pytest tests/test_llm.py tests/test_embeddings.py tests/test_labeling.py -v
```

All LLM and embedding tests should pass with mocks — no real API calls needed.
