from __future__ import annotations

import os
from typing import Optional

import httpx

from empathy_engine.config.settings import get_settings
from empathy_engine.utils.errors import TTSServiceError
from empathy_engine.utils.logging import get_logger
from empathy_engine.voice.params import VocalParams


logger = get_logger(__name__)


class ElevenLabsClient:
    """Minimal ElevenLabs TTS client wrapper.

    This client is intentionally thin and focuses on:
    - taking precomputed SSML and text
    - issuing a request to ElevenLabs' text-to-speech endpoint
    - returning raw audio bytes

    Mapping of fine-grained vocal parameters to ElevenLabs-specific controls
    (e.g. stability, style) can be iterated on separately.
    """

    def __init__(self) -> None:
        settings = get_settings()
        # Accept both prefixed and bare env var names for convenience.
        api_key = settings.elevenlabs_api_key or os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise TTSServiceError(
                "ElevenLabs API key is not configured. "
                "Set EMPATHY_ELEVENLABS_API_KEY or ELEVENLABS_API_KEY in your environment."
            )

        self._api_key = api_key
        self._base_url = settings.elevenlabs_base_url.rstrip("/")
        self._default_voice_id = settings.elevenlabs_default_voice_id
        self._model_id = settings.elevenlabs_model_id
        self._timeout = settings.tts_timeout_seconds

    async def synthesize(
        self,
        text: str,
        ssml: str,
        voice_id: Optional[str],
        audio_format: str,
        params: VocalParams,
    ) -> bytes:
        """Synthesize speech and return audio bytes.

        ElevenLabs does not natively consume SSML, but we keep it in the signature
        so the interface can support engines that do. For ElevenLabs specifically
        we send the plain text and approximate expressiveness via its settings.
        """

        vid = voice_id or self._default_voice_id
        url = f"{self._base_url}/v1/text-to-speech/{vid}"

        # Map our vocal params into ElevenLabs voice settings with enhanced responsiveness.
        # More dramatic mappings to make emotional differences more noticeable.
        
        # Stability: lower for high pitch variation (more expressive), higher for neutral
        # Range: 0.1 (very expressive) to 0.9 (very stable)
        pitch_factor = abs(params.pitch)
        stability = max(0.1, min(0.9, 0.7 - pitch_factor * 1.5))  # More responsive
        
        # Style: maps rate to expressiveness
        # Higher rate = higher style (more expressive)
        # Range: 0.0 (monotone) to 1.0 (very expressive)
        rate_offset = params.rate - 1.0
        style = max(0.0, min(1.0, 0.4 + rate_offset * 1.2))  # More dramatic scaling
        
        # Similarity boost: increase for more emotional content
        # Higher volume or pitch variation suggests more expressiveness
        emotional_intensity = min(1.0, (abs(params.pitch) * 2.0 + abs(params.volume_db) / 10.0) / 2.0)
        similarity_boost = max(0.5, min(1.0, 0.7 + emotional_intensity * 0.3))

        payload = {
            "text": text,
            "model_id": self._model_id,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
                "style": style,
                "use_speaker_boost": True,
            },
        }

        headers = {
            "xi-api-key": self._api_key,
            "Accept": f"audio/{'mpeg' if audio_format == 'mp3' else 'wav'}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:  # pragma: no cover - network layer
            msg = f"TTS HTTP error: {exc}"
            logger.exception(msg)
            raise TTSServiceError(msg) from exc

        if response.status_code >= 400:
            msg = f"TTS service error {response.status_code}: {response.text}"
            logger.error(msg)
            raise TTSServiceError(msg)

        return response.content

