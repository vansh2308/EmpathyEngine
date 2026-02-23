from __future__ import annotations

import re
from typing import List, Optional

from empathy_engine.nlp.emotion_model import EmotionPrediction


# Strong positive emotion keywords
JOY_KEYWORDS = {
    "fantastic", "amazing", "wonderful", "awesome", "great", "excellent",
    "incredible", "brilliant", "perfect", "love", "loved", "adore", "adored",
    "joy", "joyful", "happy", "happiness", "delighted", "thrilled", "ecstatic",
    "excited", "excitement", "celebrate", "celebration", "win", "won", "success",
    "successful", "achievement", "achieved", "got the job", "got it", "yes!"
}

EXCITEMENT_KEYWORDS = {
    "excited", "excitement", "thrilled", "pumped", "can't wait", "looking forward",
    "eager", "enthusiastic", "motivated", "motivation", "inspired", "inspiration"
}

# Strong negative emotion keywords
ANGER_KEYWORDS = {
    "angry", "anger", "mad", "furious", "rage", "rage", "hate", "hated",
    "annoyed", "frustrated", "frustration", "irritated", "outraged",
    "serious?", "lost the", "again?!", "what?!", "how dare", "unacceptable"
}

SADNESS_KEYWORDS = {
    "sad", "sadness", "down", "depressed", "depression", "upset", "disappointed",
    "disappointment", "unhappy", "miserable", "hopeless", "defeated", "failed",
    "nothing seems", "not working", "can't", "cannot", "won't work"
}

FEAR_KEYWORDS = {
    "worried", "worry", "anxious", "anxiety", "afraid", "fear", "scared",
    "nervous", "panic", "terrified", "dread", "concerned", "apprehensive"
}

SURPRISE_KEYWORDS = {
    "surprised", "surprise", "shocked", "shock", "unexpected", "wow", "whoa",
    "really?!", "no way", "can't believe"
}

# Positive sentiment patterns
POSITIVE_PATTERNS = [
    r"\b(got|got the|received|achieved|won|succeeded)\b",
    r"\b(fantastic|amazing|wonderful|awesome|great|excellent|incredible)\b",
    r"\b(love|adore|enjoy|delight)\b",
]

# Negative sentiment patterns
NEGATIVE_PATTERNS = [
    r"\b(lost|failed|missed|disappointed|upset|sad|down)\b",
    r"\b(serious\?|again\?!|what\?!|how dare)\b",
    r"\b(nothing|not working|can't|cannot|won't work)\b",
]


def _contains_keywords(text: str, keywords: set[str]) -> bool:
    """Check if text contains any of the given keywords."""
    lowered = text.lower()
    return any(kw in lowered for kw in keywords)


def _matches_patterns(text: str, patterns: list[str]) -> bool:
    """Check if text matches any of the given regex patterns."""
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def correct_emotion_prediction(
    text: str,
    predictions: List[EmotionPrediction],
    min_confidence: float = 0.15,
) -> Optional[EmotionPrediction]:
    """Post-process emotion predictions with rule-based corrections.
    
    Applies keyword-based corrections for obvious cases where the model
    might be wrong, and filters out very low-confidence predictions.
    
    Args:
        text: Input text
        predictions: Raw emotion predictions from the model
        min_confidence: Minimum confidence threshold (default 0.15)
    
    Returns:
        Corrected primary emotion prediction, or None if no valid prediction
    """
    if not predictions:
        return None
    
    # Sort by score
    sorted_preds = sorted(predictions, key=lambda e: e.score, reverse=True)
    top_pred = sorted_preds[0]
    
    # If top prediction is very low confidence, try keyword-based fallback
    if top_pred.score < min_confidence:
        # Try keyword-based detection
        if _contains_keywords(text, JOY_KEYWORDS) or _matches_patterns(text, POSITIVE_PATTERNS):
            # Look for joy or excitement in predictions
            for pred in sorted_preds:
                if pred.label.lower() in {"joy", "excitement", "optimism", "love"}:
                    return EmotionPrediction(label="joy", score=max(0.4, pred.score))
            # If not found, return joy anyway
            return EmotionPrediction(label="joy", score=0.5)
        
        if _contains_keywords(text, ANGER_KEYWORDS):
            for pred in sorted_preds:
                if pred.label.lower() in {"anger", "disgust", "annoyance"}:
                    return EmotionPrediction(label="anger", score=max(0.4, pred.score))
            return EmotionPrediction(label="anger", score=0.5)
        
        if _contains_keywords(text, SADNESS_KEYWORDS):
            for pred in sorted_preds:
                if pred.label.lower() in {"sadness", "disappointment", "pessimism"}:
                    return EmotionPrediction(label="sadness", score=max(0.4, pred.score))
            return EmotionPrediction(label="sadness", score=0.5)
        
        if _contains_keywords(text, FEAR_KEYWORDS):
            for pred in sorted_preds:
                if pred.label.lower() in {"fear", "nervousness", "anxiety"}:
                    return EmotionPrediction(label="fear", score=max(0.4, pred.score))
            return EmotionPrediction(label="fear", score=0.5)
        
        if _contains_keywords(text, EXCITEMENT_KEYWORDS):
            for pred in sorted_preds:
                if pred.label.lower() in {"excitement", "joy", "optimism"}:
                    return EmotionPrediction(label="excitement", score=max(0.4, pred.score))
            return EmotionPrediction(label="excitement", score=0.5)
    
    # Apply corrections for obvious misclassifications
    text_lower = text.lower()
    top_label = top_pred.label.lower()
    
    # "fantastic!!!" or similar should be joy/excitement, not optimism
    if top_label == "optimism" and (
        _contains_keywords(text, JOY_KEYWORDS) or
        _contains_keywords(text, EXCITEMENT_KEYWORDS) or
        "!!!" in text or "!!" in text
    ):
        # Find joy or excitement in predictions
        for pred in sorted_preds:
            if pred.label.lower() in {"joy", "excitement"}:
                return EmotionPrediction(
                    label=pred.label,
                    score=max(top_pred.score, pred.score + 0.1)
                )
        # Override to joy if found in keywords
        if _contains_keywords(text, JOY_KEYWORDS):
            return EmotionPrediction(label="joy", score=max(0.5, top_pred.score + 0.1))
        if _contains_keywords(text, EXCITEMENT_KEYWORDS):
            return EmotionPrediction(label="excitement", score=max(0.5, top_pred.score + 0.1))
    
    # "Are you serious? You lost the files again?!" should be anger, not joy
    if top_label == "joy" and (
        _contains_keywords(text, ANGER_KEYWORDS) or
        _matches_patterns(text, NEGATIVE_PATTERNS)
    ):
        for pred in sorted_preds:
            if pred.label.lower() in {"anger", "annoyance", "disgust"}:
                return EmotionPrediction(
                    label=pred.label,
                    score=max(top_pred.score, pred.score + 0.15)
                )
        return EmotionPrediction(label="anger", score=max(0.5, top_pred.score + 0.1))
    
    # Positive statements about motivation/excitement should not be anger
    if top_label == "anger" and (
        "motivated" in text_lower or
        "excited about" in text_lower or
        "looking forward" in text_lower or
        _contains_keywords(text, EXCITEMENT_KEYWORDS)
    ):
        for pred in sorted_preds:
            if pred.label.lower() in {"excitement", "joy", "optimism"}:
                return EmotionPrediction(
                    label=pred.label,
                    score=max(top_pred.score, pred.score + 0.15)
                )
        return EmotionPrediction(label="excitement", score=max(0.5, top_pred.score + 0.1))
    
    # "worried about" should be fear, not anger
    if top_label == "anger" and _contains_keywords(text, FEAR_KEYWORDS):
        for pred in sorted_preds:
            if pred.label.lower() in {"fear", "nervousness", "anxiety"}:
                return EmotionPrediction(
                    label=pred.label,
                    score=max(top_pred.score, pred.score + 0.1)
                )
        return EmotionPrediction(label="fear", score=max(0.4, top_pred.score))
    
    # Return the top prediction if no corrections needed
    return top_pred


def filter_and_correct_emotions(
    text: str,
    predictions: List[EmotionPrediction],
    min_confidence: float = 0.15,
) -> List[EmotionPrediction]:
    """Filter and correct emotion predictions, returning a corrected list.
    
    This function:
    1. Applies rule-based corrections
    2. Filters very low-confidence predictions
    3. Returns corrected predictions sorted by score
    """
    if not predictions:
        return []
    
    # Get corrected primary
    corrected_primary = correct_emotion_prediction(text, predictions, min_confidence)
    
    if corrected_primary is None:
        return []
    
    # Build result list with corrected primary first
    result = [corrected_primary]
    
    # Add other predictions that are different from corrected primary
    for pred in sorted(predictions, key=lambda e: e.score, reverse=True):
        if pred.label.lower() != corrected_primary.label.lower() and pred.score >= min_confidence:
            result.append(pred)
    
    return result
