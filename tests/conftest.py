"""Shared test fixtures and configuration."""
import pytest
import numpy as np


@pytest.fixture
def simple_prompts():
    return ["What is 2+2?", "Explain quantum physics.", "Write a haiku."]


@pytest.fixture
def simple_responses():
    return ["4", "Quantum physics is...", "Old silent pond..."]


@pytest.fixture
def random_embeddings():
    rng = np.random.default_rng(42)
    return rng.standard_normal((10, 16)).astype(np.float32)
