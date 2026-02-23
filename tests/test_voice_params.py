from __future__ import annotations

from empathy_engine.voice.params import NEUTRAL_PARAMS, get_base_params_for_emotion, scale_params


def test_scale_params_moves_towards_neutral() -> None:
    joy = get_base_params_for_emotion("joy")
    scaled_low = scale_params(joy, 0.2)
    scaled_high = scale_params(joy, 0.8)

    assert abs(scaled_low.rate - NEUTRAL_PARAMS.rate) < abs(scaled_high.rate - NEUTRAL_PARAMS.rate)

