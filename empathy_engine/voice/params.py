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
# These values are intentionally amplified so emotional differences are clearly audible.
BASE_EMOTION_PARAMS: Dict[str, VocalParams] = {
    # High energy positive - very pronounced
    "joy": _vp(rate=2.0, pitch=5.0, volume_db=10.0, pause_s=0.08, tremor=0.05, breathiness=0.05, resonance=0.85),
    "excitement": _vp(rate=2.2, pitch=6.0, volume_db=12.0, pause_s=0.06, tremor=0.08, breathiness=0.05, resonance=0.9),
    "surprise": _vp(rate=2.1, pitch=7.5, volume_db=11.0, pause_s=0.07, tremor=0.06, resonance=0.9),
    "love": _vp(rate=1.3, pitch=3.0, volume_db=6.0, pause_s=0.6, breathiness=0.6, resonance=0.85),
    "optimism": _vp(rate=1.7, pitch=3.5, volume_db=8.0, pause_s=0.18, resonance=0.8),
    "trust": _vp(rate=1.15, pitch=1.5, volume_db=4.0, pause_s=0.35, resonance=0.8),
    # Negative - greatly amplified
    # Sadness: very slow, very low pitch, much quieter, long pauses, lots of breathiness/tremor
    "sadness": _vp(rate=0.18, pitch=-7.0, volume_db=-28.0, pause_s=1.8, breathiness=1.0, resonance=0.06, tremor=0.75),
    "fear": _vp(rate=1.2, pitch=2.0, volume_db=4.0, pause_s=0.6, tremor=0.6, breathiness=0.35, resonance=0.45),
    "anger": _vp(rate=2.3, pitch=7.0, volume_db=14.0, pause_s=0.05, tremor=0.45, resonance=0.95),
    "disgust": _vp(rate=0.7, pitch=-2.5, volume_db=-2.0, pause_s=0.5, breathiness=0.25, resonance=0.25),
    "pessimism": _vp(rate=0.6, pitch=-3.0, volume_db=-8.0, pause_s=0.9, resonance=0.2, breathiness=0.35),
    # Low-arousal but distinct
    "boredom": _vp(rate=0.45, pitch=-2.0, volume_db=-10.0, pause_s=1.0, resonance=0.15),
    "calm": _vp(rate=0.85, pitch=-1.0, volume_db=-4.0, pause_s=0.8, resonance=0.75),
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

    # Allow a bit more headroom for scaling so very intense text can push
    # parameters beyond their base presets.
    i = max(0.0, min(1.25, intensity))

    # Start boosting earlier and stronger so even moderate intensity yields
    # clearly audible differences. Boost begins at i>0.3 and scales strongly.
    if i > 0.3:
        boost = 1.0 + (i - 0.3) * 2.0
    else:
        boost = 1.0

    def lerp(neutral_value: float, target_value: float) -> float:
        # Increase amplification for downward shifts (e.g., lower pitch/volume for sadness)
        # so negative emotions are more pronounced.
        negative_amplify = 1.8 if target_value < neutral_value else 1.0
        base_result = neutral_value + i * (target_value - neutral_value) * negative_amplify

        # Apply boost only when moving away from neutral
        if abs(target_value - neutral_value) > 0.01:
            if target_value > neutral_value:
                return neutral_value + (base_result - neutral_value) * boost
            else:
                return neutral_value - (neutral_value - base_result) * boost
        return base_result

    # Tremor and breathiness should scale non-linearly so even medium intensity
    # can result in noticeable voice texture changes.
    def textured(neutral_value: float, target_value: float) -> float:
        # Use a concave curve so values ramp up earlier: x^(0.7)
        curve = i ** 0.7
        base = neutral_value + curve * (target_value - neutral_value)
        # Give a small extra push when target is larger than neutral
        if target_value > neutral_value:
            return neutral_value + (base - neutral_value) * (1.6 if boost > 1.0 else 1.25)
        else:
            return neutral_value - (neutral_value - base) * (1.9 if boost > 1.0 else 1.4)

    return VocalParams(
        rate=lerp(NEUTRAL_PARAMS.rate, base.rate),
        pitch=lerp(NEUTRAL_PARAMS.pitch, base.pitch),
        volume_db=lerp(NEUTRAL_PARAMS.volume_db, base.volume_db),
        pause_s=lerp(NEUTRAL_PARAMS.pause_s, base.pause_s),
        tremor=textured(NEUTRAL_PARAMS.tremor, base.tremor),
        breathiness=textured(NEUTRAL_PARAMS.breathiness, base.breathiness),
        resonance=lerp(NEUTRAL_PARAMS.resonance, base.resonance),
    )

