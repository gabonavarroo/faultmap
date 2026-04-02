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
