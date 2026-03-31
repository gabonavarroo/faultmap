"""Custom exceptions for faultmap."""


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
