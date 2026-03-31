import numpy as np
import pytest
from faultmap.utils import (
    cosine_similarity_matrix,
    cosine_similarity_pairs,
    validate_inputs,
    batch_items,
)
from faultmap.exceptions import ConfigurationError


class TestCosineSimilarityMatrix:
    def test_identical_vectors(self):
        a = np.array([[1.0, 0.0, 0.0]])
        result = cosine_similarity_matrix(a, a)
        assert result.shape == (1, 1)
        assert np.isclose(result[0, 0], 1.0)

    def test_orthogonal_vectors(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 1.0]])
        result = cosine_similarity_matrix(a, b)
        assert np.isclose(result[0, 0], 0.0, atol=1e-6)

    def test_antiparallel_vectors(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[-1.0, 0.0]])
        result = cosine_similarity_matrix(a, b)
        assert np.isclose(result[0, 0], -1.0)

    def test_batch_shape(self):
        a = np.random.randn(5, 10)
        b = np.random.randn(3, 10)
        result = cosine_similarity_matrix(a, b)
        assert result.shape == (5, 3)


class TestCosineSimilarityPairs:
    def test_identical(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = cosine_similarity_pairs(a, a)
        assert np.allclose(result, [1.0, 1.0], atol=1e-6)

    def test_orthogonal(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 1.0]])
        result = cosine_similarity_pairs(a, b)
        assert np.isclose(result[0], 0.0, atol=1e-6)


class TestValidateInputs:
    def test_empty_prompts(self):
        with pytest.raises(ConfigurationError, match="non-empty"):
            validate_inputs([], [], None, None)

    def test_length_mismatch(self):
        with pytest.raises(ConfigurationError, match="equal length"):
            validate_inputs(["a"], ["b", "c"], None, None)

    def test_score_out_of_range(self):
        with pytest.raises(ConfigurationError, match="out of range"):
            validate_inputs(["a"], ["b"], [1.5], None)

    def test_score_not_numeric(self):
        with pytest.raises(ConfigurationError, match="numeric"):
            validate_inputs(["a"], ["b"], ["bad"], None)

    def test_valid_inputs(self):
        validate_inputs(["a", "b"], ["c", "d"], [0.5, 0.8], None)

    def test_valid_with_references(self):
        validate_inputs(["a"], ["b"], None, ["c"])


class TestBatchItems:
    def test_even_split(self):
        result = batch_items([1, 2, 3, 4], 2)
        assert result == [[1, 2], [3, 4]]

    def test_uneven_split(self):
        result = batch_items([1, 2, 3], 2)
        assert result == [[1, 2], [3]]

    def test_empty(self):
        assert batch_items([], 5) == []
