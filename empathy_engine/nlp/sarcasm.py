from __future__ import annotations

import re
from typing import Iterable, Sequence

from empathy_engine.nlp.emotion_model import EmotionPrediction
from empathy_engine.nlp.spacy_utils import contains_contrastive_conjunction, has_negation


POSITIVE_HINTS = {"great", "awesome", "amazing", "fantastic", "wonderful", "nice"}
NEGATIVE_HINTS = {"terrible", "awful", "horrible", "bad", "sucks", "worst"}


def _contains_phrase(text: str, phrases: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(p in lowered for p in phrases)


def estimate_sarcasm_score(
    text: str,
    emotions: Sequence[EmotionPrediction],
) -> float:
    """Heuristic sarcasm / irony score in [0, 1].

    Uses cues:
    - contrastive conjunctions (but, however, ...)
    - sentiment inconsistency between words and detected emotion
    - flat punctuation for ostensibly positive phrases
    """

    if not text.strip():
        return 0.0

    sarcasm = 0.0
    lowered = text.lower()

    # Contrastive conjunctions strongly suggest a flip.
    if contains_contrastive_conjunction(text):
        sarcasm += 0.4

    # Explicit sarcastic phrases.
    if _contains_phrase(
        lowered,
        {"yeah right", "as if", "sure you are", "of course you did"},
    ):
        sarcasm += 0.5

    # Positive phrase + negative hints or vice versa.
    has_positive_word = _contains_phrase(lowered, POSITIVE_HINTS)
    has_negative_word = _contains_phrase(lowered, NEGATIVE_HINTS)
    negation_present = has_negation(text)

    primary = max(emotions, key=lambda e: e.score) if emotions else None
    primary_label = primary.label.lower() if primary else ""

    if has_positive_word and (primary_label in {"anger", "sadness", "disgust"} or has_negative_word or negation_present):
        sarcasm += 0.3
    if has_negative_word and primary_label in {"joy", "love"}:
        sarcasm += 0.3

    # Flat punctuation on very positive phrases (e.g. "Great." vs "Great!")
    if has_positive_word and not re.search(r"[!?]", text):
        sarcasm += 0.1

    return max(0.0, min(1.0, sarcasm))

