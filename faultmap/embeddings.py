from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Literal

import numpy as np

from .exceptions import EmbeddingError

logger = logging.getLogger(__name__)

EmbeddingUsage = Literal["generic", "query", "document"]


class Embedder(ABC):
    """Abstract base class for text embedding."""

    @abstractmethod
    def embed(
        self,
        texts: list[str],
        *,
        usage: EmbeddingUsage = "generic",
    ) -> np.ndarray:
        """Embed texts. Returns shape (len(texts), embedding_dim)."""
        ...

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        """Embed texts as search queries when supported by the model."""
        return self.embed(texts, usage="query")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed texts as documents/passages when supported by the model."""
        return self.embed(texts, usage="document")

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

    def embed(
        self,
        texts: list[str],
        *,
        usage: EmbeddingUsage = "generic",
    ) -> np.ndarray:
        if self._model is None:
            self._load_model()
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        encode_kwargs = {
            "batch_size": self.batch_size,
            "show_progress_bar": len(texts) > 100,
            "convert_to_numpy": True,
        }

        # sentence-transformers v5 exposes explicit query/document helpers.
        # Older versions or symmetric models safely fall back to encode().
        if usage == "query" and hasattr(self._model, "encode_query"):
            embeddings = self._model.encode_query(texts, **encode_kwargs)
        elif usage == "document" and hasattr(self._model, "encode_document"):
            embeddings = self._model.encode_document(texts, **encode_kwargs)
        else:
            embeddings = self._model.encode(texts, **encode_kwargs)
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
    - Very long texts → truncated by characters before API call
    - Rate limiting → litellm handles retries internally
    - API error → wrap in EmbeddingError
    """

    DEFAULT_MAX_TEXT_CHARS = 2000

    def __init__(
        self,
        model_name: str,
        batch_size: int = 128,
        max_text_chars: int | None = DEFAULT_MAX_TEXT_CHARS,
        request_kwargs: Mapping[str, object] | None = None,
        usage_request_kwargs: Mapping[EmbeddingUsage, Mapping[str, object]] | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_text_chars = max_text_chars
        self.request_kwargs = dict(request_kwargs or {})
        self.usage_request_kwargs = self._merge_usage_request_kwargs(
            usage_request_kwargs
        )
        self._dimension: int | None = None

    def _merge_usage_request_kwargs(
        self,
        usage_request_kwargs: Mapping[EmbeddingUsage, Mapping[str, object]] | None,
    ) -> dict[EmbeddingUsage, dict[str, object]]:
        merged = self._default_usage_request_kwargs()
        if usage_request_kwargs is not None:
            for usage, kwargs in usage_request_kwargs.items():
                merged.setdefault(usage, {})
                merged[usage].update(dict(kwargs))
        return merged

    def _default_usage_request_kwargs(self) -> dict[EmbeddingUsage, dict[str, object]]:
        model_name = self.model_name.lower()
        if "nv-embedqa" in model_name:
            return {
                "query": {"input_type": "query"},
                "document": {"input_type": "passage"},
            }
        return {}

    def _build_embedding_request(
        self,
        texts: list[str],
        *,
        usage: EmbeddingUsage,
    ) -> dict[str, object]:
        request: dict[str, object] = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float",
        }
        request.update(self.request_kwargs)
        request.update(self.usage_request_kwargs.get(usage, {}))
        return request

    def _truncate_texts(self, texts: list[str]) -> list[str]:
        if self.max_text_chars is None:
            return texts

        truncated = [text[: self.max_text_chars] for text in texts]
        num_truncated = sum(
            1 for original, shortened in zip(texts, truncated) if len(original) != len(shortened)
        )
        if num_truncated:
            logger.warning(
                "Truncated %s embedding input(s) to %s chars for model %s",
                num_truncated,
                self.max_text_chars,
                self.model_name,
            )
        return truncated

    def _probe_dimension(self, *, usage: EmbeddingUsage) -> int:
        import litellm

        litellm.telemetry = False

        try:
            probe = litellm.embedding(
                **self._build_embedding_request(["dimension probe"], usage=usage)
            )
        except Exception as e:
            raise EmbeddingError(f"API embedding dimension probe failed: {e}") from e
        self._dimension = len(probe.data[0]["embedding"])
        return self._dimension

    def embed(
        self,
        texts: list[str],
        *,
        usage: EmbeddingUsage = "generic",
    ) -> np.ndarray:
        import litellm

        litellm.telemetry = False
        if not texts:
            if self._dimension is not None:
                return np.empty((0, self._dimension), dtype=np.float32)
            return np.empty((0, self._probe_dimension(usage=usage)), dtype=np.float32)

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            safe_batch = self._truncate_texts(batch)
            try:
                response = litellm.embedding(
                    **self._build_embedding_request(safe_batch, usage=usage)
                )
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
            self._probe_dimension(usage="generic")
        return self._dimension


def get_embedder(
    model_name: str,
    *,
    api_max_text_chars: int | None = APIEmbedder.DEFAULT_MAX_TEXT_CHARS,
    api_request_kwargs: Mapping[str, object] | None = None,
    api_usage_request_kwargs: Mapping[EmbeddingUsage, Mapping[str, object]] | None = None,
) -> Embedder:
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

    return APIEmbedder(
        model_name,
        max_text_chars=api_max_text_chars,
        request_kwargs=api_request_kwargs,
        usage_request_kwargs=api_usage_request_kwargs,
    )
