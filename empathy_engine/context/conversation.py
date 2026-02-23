from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from empathy_engine.context.embedding import embed_text_with_emotions
from empathy_engine.context.vector_store import FaissVectorStore, StoredItem
from empathy_engine.nlp.emotion_model import EmotionPrediction
from empathy_engine.voice.params import VocalParams, NEUTRAL_PARAMS, get_base_params_for_emotion, scale_params


@dataclass
class EmotionalContext:
    dominant_emotion: str
    intensity: float
    params: VocalParams
    history: List[StoredItem]


class ConversationContextManager:
    """High-level API for storing and retrieving emotional context."""

    def __init__(self) -> None:
        self._store = FaissVectorStore()

    @property
    def is_ready(self) -> bool:
        return self._store.is_ready

    def update_context(
        self,
        session_id: Optional[str],
        text: str,
        emotions: List[EmotionPrediction],
        params: VocalParams,
    ) -> None:
        embedding = embed_text_with_emotions(text, emotions)
        primary = max(emotions, key=lambda e: e.score) if emotions else None
        meta: Dict[str, Any] = {
            "emotion_primary": primary.label if primary else "unknown",
            "emotion_scores": [{ "label": e.label, "score": e.score } for e in emotions],
            "vocal_params": {
                "rate": params.rate,
                "pitch": params.pitch,
                "volume_db": params.volume_db,
                "pause_s": params.pause_s,
                "tremor": params.tremor,
                "breathiness": params.breathiness,
                "resonance": params.resonance,
            },
        }
        self._store.add_item(
            embedding=embedding,
            text=text,
            session_id=session_id,
            meta=meta,
        )

    def get_recent_emotional_context(
        self,
        session_id: Optional[str],
        limit: int = 5,
    ) -> Optional[EmotionalContext]:
        if not session_id:
            return None
        history = self._store.get_recent_for_session(session_id, limit=limit)
        if not history:
            return None

        # Approximate dominant emotion and intensity from history.
        label_scores: Dict[str, float] = {}
        for item in history:
            for e in item.meta.get("emotion_scores", []):
                label = e["label"]
                score = float(e["score"])
                label_scores[label] = label_scores.get(label, 0.0) + score

        if not label_scores:
            return EmotionalContext(
                dominant_emotion="neutral",
                intensity=0.3,
                params=NEUTRAL_PARAMS,
                history=history,
            )

        dominant_label, total_score = max(label_scores.items(), key=lambda kv: kv[1])
        max_possible = len(history)  # rough upper bound
        intensity = min(1.0, total_score / max_possible)

        base = get_base_params_for_emotion(dominant_label)
        params = scale_params(base, intensity)

        return EmotionalContext(
            dominant_emotion=dominant_label,
            intensity=intensity,
            params=params,
            history=history,
        )

