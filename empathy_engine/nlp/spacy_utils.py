from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, Set

import spacy
from spacy.language import Language

from empathy_engine.config.settings import get_settings
from empathy_engine.utils.logging import get_logger


logger = get_logger(__name__)

INTENSITY_ADVERBS: Set[str] = {
    "very",
    "extremely",
    "really",
    "so",
    "incredibly",
    "absolutely",
    "totally",
    "utterly",
    "completely",
}

CONTRASTIVE_CONJUNCTIONS: Set[str] = {
    "but",
    "however",
    "though",
    "although",
    "yet",
    "nevertheless",
}


@lru_cache(maxsize=1)
def get_spacy_nlp() -> Language:
    """Load and cache the spaCy language model."""

    settings = get_settings()
    model_name = settings.spacy_model_name

    try:
        logger.info("Loading spaCy model: %s", model_name)
        return spacy.load(model_name)
    except Exception:  # noqa: BLE001
        # Fallback to a blank English model if full model is unavailable.
        logger.warning(
            "Failed to load spaCy model '%s'. Falling back to blank 'en'.", model_name
        )
        return spacy.blank("en")


def count_exclamation_marks(text: str) -> int:
    return text.count("!")


def capital_words_ratio(text: str) -> float:
    words = re.findall(r"\b\w+\b", text)
    if not words:
        return 0.0
    capped = [w for w in words if len(w) > 1 and w.isupper()]
    return len(capped) / len(words)


def intensity_adverbs_score(text: str) -> float:
    """Return a normalized score based on intensity adverbs in the text."""

    doc = get_spacy_nlp()(text)
    matches: Iterable[str] = (
        token.lemma_.lower() if token.lemma_ else token.text.lower()
        for token in doc
        if token.pos_ in {"ADV", "ADJ"} or token.text.lower() in INTENSITY_ADVERBS
    )
    count = sum(1 for w in matches if w in INTENSITY_ADVERBS)
    # Simple normalization: more than 4 intensifiers saturates the score.
    if count == 0:
        return 0.0
    return min(1.0, count / 4.0)


def contains_contrastive_conjunction(text: str) -> bool:
    doc = get_spacy_nlp()(text)
    return any(token.text.lower() in CONTRASTIVE_CONJUNCTIONS for token in doc)


def has_negation(text: str) -> bool:
    doc = get_spacy_nlp()(text)
    return any(token.dep_ == "neg" or token.text.lower() in {"not", "never"} for token in doc)

