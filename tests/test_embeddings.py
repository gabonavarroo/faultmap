from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from faultmap.embeddings import APIEmbedder, LocalEmbedder, get_embedder
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
        embedder.batch_size = 64
        embedder.device = None
        embedder._dimension = 64
        embedder._model = MagicMock()
        result = embedder.embed([])
        assert result.shape == (0, 64)

    def test_query_usage_prefers_encode_query(self):
        embedder = LocalEmbedder.__new__(LocalEmbedder)
        embedder.batch_size = 64
        embedder.device = None
        embedder._dimension = 2
        embedder._model = MagicMock()
        embedder._model.encode_query.return_value = np.array([[1.0, 0.0]])

        result = embedder.embed(["test"], usage="query")

        assert result.shape == (1, 2)
        embedder._model.encode_query.assert_called_once()
        embedder._model.encode.assert_not_called()

    def test_document_usage_falls_back_to_encode(self):
        embedder = LocalEmbedder.__new__(LocalEmbedder)
        embedder.batch_size = 64
        embedder.device = None
        embedder._dimension = 2
        embedder._model = MagicMock(spec=["encode"])
        embedder._model.encode.return_value = np.array([[0.0, 1.0]])

        result = embedder.embed(["test"], usage="document")

        assert result.shape == (1, 2)
        embedder._model.encode.assert_called_once()


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

    def test_symmetric_models_ignore_usage_by_default(self):
        embedder = APIEmbedder("text-embedding-3-small")

        mock_resp = MagicMock()
        mock_resp.data = [{"embedding": [0.0, 1.0], "index": 0}]

        with patch("litellm.embedding", return_value=mock_resp) as mock:
            embedder.embed(["hello"], usage="query")

        assert mock.call_args.kwargs["encoding_format"] == "float"
        assert "input_type" not in mock.call_args.kwargs

    def test_nvidia_models_get_role_specific_input_type(self):
        embedder = APIEmbedder("nvidia_nim/nvidia/nv-embedqa-e5-v5")

        mock_resp = MagicMock()
        mock_resp.data = [{"embedding": [0.0, 1.0], "index": 0}]

        with patch("litellm.embedding", return_value=mock_resp) as mock:
            embedder.embed(["hello"], usage="document")

        assert mock.call_args.kwargs["input_type"] == "passage"

    def test_user_usage_kwargs_override_model_defaults(self):
        embedder = APIEmbedder(
            "nvidia_nim/nvidia/nv-embedqa-e5-v5",
            usage_request_kwargs={"document": {"input_type": "document"}},
        )

        mock_resp = MagicMock()
        mock_resp.data = [{"embedding": [0.0, 1.0], "index": 0}]

        with patch("litellm.embedding", return_value=mock_resp) as mock:
            embedder.embed(["hello"], usage="document")

        assert mock.call_args.kwargs["input_type"] == "document"

    def test_long_texts_are_truncated_before_api_call(self):
        embedder = APIEmbedder("text-embedding-3-small", max_text_chars=10)
        long_text = "x" * 25

        mock_resp = MagicMock()
        mock_resp.data = [{"embedding": [0.0, 1.0], "index": 0}]

        with patch("litellm.embedding", return_value=mock_resp) as mock:
            embedder.embed([long_text])

        assert mock.call_args.kwargs["input"] == ["x" * 10]

    def test_truncation_can_be_disabled(self):
        embedder = APIEmbedder("text-embedding-3-small", max_text_chars=None)
        long_text = "x" * 25

        mock_resp = MagicMock()
        mock_resp.data = [{"embedding": [0.0, 1.0], "index": 0}]

        with patch("litellm.embedding", return_value=mock_resp) as mock:
            embedder.embed([long_text])

        assert mock.call_args.kwargs["input"] == [long_text]


class TestGetEmbedder:
    def test_local_model_names(self):
        assert isinstance(get_embedder("all-MiniLM-L6-v2"), LocalEmbedder)
        assert isinstance(get_embedder("all-mpnet-base-v2"), LocalEmbedder)
        assert isinstance(get_embedder("bge-small-en-v1.5"), LocalEmbedder)
        assert isinstance(get_embedder("BAAI/bge-large-en"), LocalEmbedder)

    def test_api_model_names(self):
        assert isinstance(get_embedder("text-embedding-3-small"), APIEmbedder)
        assert isinstance(get_embedder("text-embedding-ada-002"), APIEmbedder)

    def test_api_kwargs_are_forwarded(self):
        embedder = get_embedder(
            "text-embedding-3-small",
            api_max_text_chars=500,
            api_request_kwargs={"dimensions": 512},
            api_usage_request_kwargs={"query": {"input_type": "query"}},
        )

        assert isinstance(embedder, APIEmbedder)
        assert embedder.max_text_chars == 500
        assert embedder.request_kwargs == {"dimensions": 512}
        assert embedder.usage_request_kwargs["query"]["input_type"] == "query"
