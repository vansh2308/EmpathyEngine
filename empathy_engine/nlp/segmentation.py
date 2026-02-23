from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

from empathy_engine.nlp.emotion_model import EmotionPrediction
from empathy_engine.nlp.spacy_utils import get_spacy_nlp


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using spaCy, with a simple fallback."""

    if not text.strip():
        return []

    nlp = get_spacy_nlp()
    if not nlp.has_pipe("sentencizer") and "parser" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences or [text.strip()]


@dataclass
class EmotionalArc:
    per_sentence_emotions: List[List[EmotionPrediction]]
    has_arc: bool
    arc_type: str | None = None


def detect_emotional_arc(
    text: str,
    detect_emotions: Callable[[str], Sequence[EmotionPrediction]],
) -> EmotionalArc:
    """Detect a simple emotional arc across sentences.

    This is a first-pass heuristic implementation that:
    - detects dominant emotion per sentence
    - checks if there is a change in dominant emotion or a clear intensity trend
    """

    sentences = split_sentences(text)
    per_sentence: List[List[EmotionPrediction]] = []
    dominant_labels: List[str] = []

    for sent in sentences:
        preds = list(detect_emotions(sent))
        per_sentence.append(preds)
        if preds:
            dominant = max(preds, key=lambda p: p.score)
            dominant_labels.append(dominant.label)
        else:
            dominant_labels.append("neutral")

    has_arc = False
    arc_type: str | None = None

    if len(dominant_labels) >= 2:
        if len(set(dominant_labels)) > 1:
            has_arc = True
            arc_type = f"{dominant_labels[0]}_to_{dominant_labels[-1]}"

    return EmotionalArc(
        per_sentence_emotions=per_sentence,
        has_arc=has_arc,
        arc_type=arc_type,
    )

