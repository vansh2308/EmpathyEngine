from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class VocalParams:
    rate: float
    pitch: float
    volume_db: float
    pause_s: float
    tremor: float = 0.0
    breathiness: float = 0.0
    resonance: float = 0.0


NEUTRAL_PARAMS = VocalParams(
    rate=1.0,
    pitch=0.0,
    volume_db=0.0,
    pause_s=0.3,
    tremor=0.0,
    breathiness=0.0,
    resonance=0.5,
)


def _vp(
    rate: float,
    pitch: float,
    volume_db: float,
    pause_s: float,
    tremor: float = 0.0,
    breathiness: float = 0.0,
    resonance: float = 0.5,
) -> VocalParams:
    return VocalParams(
        rate=rate,
        pitch=pitch,
        volume_db=volume_db,
        pause_s=pause_s,
        tremor=tremor,
        breathiness=breathiness,
        resonance=resonance,
    )


# Base mappings for common emotions. Any unmapped label will fall back to NEUTRAL_PARAMS.
# These are intentionally more extreme to create noticeable emotional differences.
BASE_EMOTION_PARAMS: Dict[str, VocalParams] = {
    # High energy positive - much more dramatic
    "joy": _vp(rate=1.6, pitch=0.3, volume_db=6.0, pause_s=0.15, tremor=0.1),
    "excitement": _vp(rate=1.8, pitch=0.35, volume_db=8.0, pause_s=0.1, tremor=0.15),
    "surprise": _vp(rate=1.7, pitch=0.4, volume_db=7.0, pause_s=0.2, tremor=0.12),
    "love": _vp(rate=1.2, pitch=0.2, volume_db=4.0, pause_s=0.4, breathiness=0.25, resonance=0.7),
    "optimism": _vp(rate=1.4, pitch=0.25, volume_db=5.0, pause_s=0.25),
    "trust": _vp(rate=1.1, pitch=0.1, volume_db=2.0, pause_s=0.4, resonance=0.65),
    # Negative - more pronounced
    "sadness": _vp(rate=0.6, pitch=-0.25, volume_db=-6.0, pause_s=0.6, breathiness=0.2, resonance=0.3),
    "fear": _vp(rate=1.4, pitch=0.25, volume_db=3.0, pause_s=0.5, tremor=0.3, breathiness=0.15),
    "anger": _vp(rate=1.7, pitch=0.5, volume_db=10.0, pause_s=0.15, tremor=0.2, resonance=0.8),
    "disgust": _vp(rate=0.85, pitch=-0.15, volume_db=2.0, pause_s=0.4, breathiness=0.1),
    "pessimism": _vp(rate=0.75, pitch=-0.2, volume_db=-3.0, pause_s=0.55, resonance=0.35),
    # Lower arousal - more distinct
    "boredom": _vp(rate=0.65, pitch=-0.1, volume_db=-4.0, pause_s=0.7, resonance=0.3),
    "calm": _vp(rate=0.9, pitch=-0.05, volume_db=-2.0, pause_s=0.5, resonance=0.65),
}


def get_base_params_for_emotion(label: str) -> VocalParams:
    """Return base vocal parameters for a given emotion label."""

    key = label.lower()
    return BASE_EMOTION_PARAMS.get(key, NEUTRAL_PARAMS)


def scale_params(base: VocalParams, intensity: float) -> VocalParams:
    """Scale base params by intensity ∈ [0, 1.1] with boost for high intensity.
    
    At intensity=1.0, we reach the base value. At intensity > 0.7, we apply
    an additional boost multiplier to push beyond base values for maximum expressiveness.
    Allows slight overdrive (up to 1.1) for extremely intense text.
    """

    i = max(0.0, min(1.1, intensity))
    
    # Apply boost multiplier for high intensity (allows "overdrive" effect)
    # At intensity 0.7-1.0, we gradually increase the multiplier up to 1.3x
    if i > 0.7:
        boost = 1.0 + (i - 0.7) * 1.0  # 1.0x at 0.7, up to 1.3x at 1.0
    else:
        boost = 1.0

    def lerp(neutral_value: float, target_value: float) -> float:
        base_result = neutral_value + i * (target_value - neutral_value)
        # Apply boost only if we're moving away from neutral
        if abs(target_value - neutral_value) > 0.01:
            # Boost pushes further from neutral
            if target_value > neutral_value:
                boosted = neutral_value + (base_result - neutral_value) * boost
            else:
                boosted = neutral_value - (neutral_value - base_result) * boost
            return boosted
        return base_result

    return VocalParams(
        rate=lerp(NEUTRAL_PARAMS.rate, base.rate),
        pitch=lerp(NEUTRAL_PARAMS.pitch, base.pitch),
        volume_db=lerp(NEUTRAL_PARAMS.volume_db, base.volume_db),
        pause_s=lerp(NEUTRAL_PARAMS.pause_s, base.pause_s),
        tremor=lerp(NEUTRAL_PARAMS.tremor, base.tremor),
        breathiness=lerp(NEUTRAL_PARAMS.breathiness, base.breathiness),
        resonance=lerp(NEUTRAL_PARAMS.resonance, base.resonance),
    )

