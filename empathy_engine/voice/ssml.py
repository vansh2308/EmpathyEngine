from __future__ import annotations

from html import escape
from typing import Iterable, List

from empathy_engine.voice.params import VocalParams


def _prosody_attrs(params: VocalParams) -> str:
    rate_pct = int(params.rate * 100)
    pitch_pct = int(params.pitch * 100)
    volume_db = params.volume_db

    return f'rate="{rate_pct}%" pitch="{pitch_pct}%" volume="{volume_db:+.1f}dB"'


def build_sentence_level_ssml(
    sentences: Iterable[str],
    sentence_params: Iterable[VocalParams],
) -> str:
    """Build SSML with per-sentence prosody and dynamic pauses."""

    ssml_parts: List[str] = ["<speak>"]
    for sent, params in zip(sentences, sentence_params):
        sent_clean = escape(sent.strip())
        attrs = _prosody_attrs(params)
        ssml_parts.append(f'<p><prosody {attrs}>{sent_clean}</prosody></p>')
        # Use the sentence's pause as a break before the next one.
        ssml_parts.append(f'<break time="{params.pause_s:.2f}s"/>')
    ssml_parts.append("</speak>")
    return "".join(ssml_parts)


def build_simple_ssml(text: str, params: VocalParams) -> str:
    """Fallback: wrap full text in a single prosody block."""

    attrs = _prosody_attrs(params)
    safe_text = escape(text.strip())
    return f"<speak><p><prosody {attrs}>{safe_text}</prosody></p></speak>"

