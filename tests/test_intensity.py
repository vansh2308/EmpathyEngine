from __future__ import annotations

from empathy_engine.nlp.intensity import calculate_intensity


def test_intensity_increases_with_exclamations() -> None:
    base = calculate_intensity("this is fine", 0.5)
    excited = calculate_intensity("THIS IS FINE!!!", 0.9)
    assert excited > base

