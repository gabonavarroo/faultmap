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
