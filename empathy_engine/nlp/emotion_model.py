from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from empathy_engine.config.settings import get_settings
from empathy_engine.utils.errors import EmotionModelError
from empathy_engine.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class EmotionPrediction:
    label: str
    score: float


class EmotionDetector:
    """Wrapper around a Hugging Face emotion classification model."""

    def __init__(self) -> None:
        settings = get_settings()
        model_name = settings.emotion_model_name
        try:
            logger.info("Loading emotion model: %s", model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        except Exception as exc:  # noqa: BLE001
            msg = f"Failed to load emotion model '{model_name}': {exc}"
            logger.exception(msg)
            raise EmotionModelError(msg) from exc

        self._pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True,
        )
        self._labels = [l for l in self._pipeline.model.config.id2label.values()]
        logger.info("Emotion model loaded with labels: %s", ", ".join(self._labels))

    @property
    def labels(self) -> List[str]:
        return self._labels

    def detect(self, text: str) -> List[EmotionPrediction]:
        """Return emotion scores for a single piece of text."""

        try:
            outputs = self._pipeline(text)[0]
        except Exception as exc:  # noqa: BLE001
            msg = f"Emotion detection failed: {exc}"
            logger.exception(msg)
            raise EmotionModelError(msg) from exc

        # outputs is a list of dicts with keys: label, score
        return [
            EmotionPrediction(label=o["label"], score=float(o["score"]))
            for o in outputs
        ]

    def detect_batch(self, texts: List[str]) -> List[List[EmotionPrediction]]:
        """Return emotion scores for a batch of texts."""

        try:
            raw_outputs = self._pipeline(texts)
        except Exception as exc:  # noqa: BLE001
            msg = f"Batch emotion detection failed: {exc}"
            logger.exception(msg)
            raise EmotionModelError(msg) from exc

        results: List[List[EmotionPrediction]] = []
        for item in raw_outputs:
            preds = [
                EmotionPrediction(label=o["label"], score=float(o["score"]))
                for o in item
            ]
            results.append(preds)

        return results


@lru_cache(maxsize=1)
def get_emotion_detector() -> EmotionDetector:
    """Return a cached emotion detector instance."""

    return EmotionDetector()

