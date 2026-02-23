from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from empathy_engine.config.settings import get_settings
from empathy_engine.nlp.emotion_model import EmotionPrediction
from empathy_engine.utils.logging import get_logger


logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    settings = get_settings()
    model_name = settings.embedding_model_name
    logger.info("Loading embedding model: %s", model_name)
    return SentenceTransformer(model_name)


def _emotions_to_vector(
    emotions: Sequence[EmotionPrediction],
    label_order: Iterable[str] | None = None,
    dim: int = 32,
) -> np.ndarray:
    """Convert a set of emotions into a fixed-size vector.

    For now we simply take the top-k scores and pad/truncate to `dim`.
    This is intentionally simple; a production system could learn a more
    structured embedding over the emotion space.
    """

    if not emotions:
        return np.zeros(dim, dtype=np.float32)

    scores = sorted((e.score for e in emotions), reverse=True)
    vec = np.zeros(dim, dtype=np.float32)
    for i, s in enumerate(scores[:dim]):
        vec[i] = float(s)
    return vec


def embed_text_with_emotions(
    text: str,
    emotions: Sequence[EmotionPrediction],
) -> np.ndarray:
    """Return a combined embedding of text semantics and emotion distribution."""

    model = get_embedding_model()
    text_vec = model.encode(text, normalize_embeddings=True)

    emotion_vec = _emotions_to_vector(emotions, dim=min(32, text_vec.shape[0]))

    # Concatenate and, if necessary, truncate/pad to configured dimension.
    settings = get_settings()
    target_dim = settings.embedding_dimension

    combined = np.concatenate([text_vec, emotion_vec], axis=0)

    if combined.shape[0] > target_dim:
        combined = combined[:target_dim]
    elif combined.shape[0] < target_dim:
        pad = np.zeros(target_dim - combined.shape[0], dtype=np.float32)
        combined = np.concatenate([combined, pad], axis=0)

    return combined.astype(np.float32)

