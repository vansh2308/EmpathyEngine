from __future__ import annotations

from empathy_engine.nlp.spacy_utils import (
    capital_words_ratio,
    count_exclamation_marks,
    intensity_adverbs_score,
)


def calculate_intensity(text: str, emotion_confidence: float) -> float:
    """Compute an overall emotional intensity score in [0, 1].

    Combines multiple signals with weighted contributions:
    - number of exclamation marks (more weight)
    - ratio of capitalized words (more weight)
    - presence of intensity adverbs (high weight)
    - model emotion confidence (high weight)
    
    Uses weighted average instead of simple average to allow higher intensity values.
    """

    ex_mark_signal = min(1.0, count_exclamation_marks(text) * 0.55)  # Increased from 0.2
    caps_signal = max(0.0, min(1.0, capital_words_ratio(text) * 4.0))  # Increased from 3.0
    adverb_signal = intensity_adverbs_score(text)
    confidence_signal = max(0.0, min(1.0, emotion_confidence))

    # Weighted average: confidence and adverbs are most important
    # This allows intensity to reach higher values more easily
    raw = (
        ex_mark_signal * 0.3 +
        caps_signal * 0.2 +
        adverb_signal * 0.3 +
        confidence_signal * 0.3
    )
    
    # Allow slight overdrive (up to 1.1) for extremely intense text
    return max(0.0, min(1.1, raw))

