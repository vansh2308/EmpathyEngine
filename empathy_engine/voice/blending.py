from __future__ import annotations

from typing import List, Tuple

from empathy_engine.nlp.emotion_model import EmotionPrediction
from empathy_engine.voice.params import VocalParams


def select_primary_secondary(
    emotions: List[EmotionPrediction],
    threshold: float = 0.3,
) -> Tuple[EmotionPrediction | None, EmotionPrediction | None]:
    """Select primary and optional secondary emotions from a list of scores."""

    if not emotions:
        return None, None

    sorted_emotions = sorted(emotions, key=lambda e: e.score, reverse=True)
    primary = sorted_emotions[0]
    secondary = sorted_emotions[1] if len(sorted_emotions) > 1 else None

    if secondary and secondary.score < threshold:
        secondary = None

    return primary, secondary


def blend_voice_params(
    primary: VocalParams,
    secondary: VocalParams | None,
    blend_ratio: float = 0.7,
) -> VocalParams:
    """Blend two sets of vocal params with the given ratio.

    blend_ratio weights the primary params; (1 - blend_ratio) the secondary.
    """

    if secondary is None:
        return primary

    r = max(0.0, min(1.0, blend_ratio))
    s = 1.0 - r

    def mix(a: float, b: float) -> float:
        return r * a + s * b

    return VocalParams(
        rate=mix(primary.rate, secondary.rate),
        pitch=mix(primary.pitch, secondary.pitch),
        volume_db=mix(primary.volume_db, secondary.volume_db),
        pause_s=mix(primary.pause_s, secondary.pause_s),
        tremor=mix(primary.tremor, secondary.tremor),
        breathiness=mix(primary.breathiness, secondary.breathiness),
        resonance=mix(primary.resonance, secondary.resonance),
    )

