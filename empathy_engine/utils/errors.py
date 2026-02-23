from __future__ import annotations


class EmpathyEngineError(Exception):
    """Base class for EmpathyEngine-specific errors."""


class EmotionModelError(EmpathyEngineError):
    """Raised when the emotion model fails or is not available."""


class TTSServiceError(EmpathyEngineError):
    """Raised for errors communicating with the TTS backend."""


class VectorStoreError(EmpathyEngineError):
    """Raised when the vector store encounters an unrecoverable error."""

