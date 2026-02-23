from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class EmotionScore(BaseModel):
    label: str
    score: float


class VocalParamsSchema(BaseModel):
    rate: float = Field(..., description="Relative speaking rate multiplier (0.5–2.0).")
    pitch: float = Field(..., description="Relative pitch adjustment from baseline (-0.3–0.3).")
    volume_db: float = Field(..., description="Volume adjustment in decibels (-10 to +10).")
    pause_s: float = Field(..., description="Average pause duration between phrases, in seconds.")
    emphasis: Dict[str, float] = Field(
        default_factory=dict,
        description="Optional word-level emphasis strengths keyed by token.",
    )
    tremor: float = Field(
        0.0,
        description="Optional emotional quiver intensity (0–1).",
    )
    breathiness: float = Field(
        0.0,
        description="Optional breathiness level (0–1).",
    )
    resonance: float = Field(
        0.0,
        description="Optional resonance shift between chest/head voice (0–1).",
    )


class DebugInfo(BaseModel):
    emotions: List[EmotionScore]
    intensity: float
    primary_emotion: str
    secondary_emotion: Optional[str] = None
    vocal_params: VocalParamsSchema
    ssml: str
    context_used: Optional[dict] = None


class SynthesisRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to synthesize.")
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session identifier for conversational context.",
    )
    language: str = Field("en", description="Language code of the input text.")
    voice_id: Optional[str] = Field(
        default=None,
        description="Optional override of the default TTS voice.",
    )
    format: str = Field(
        "mp3",
        pattern="^(mp3|wav)$",
        description="Output audio format.",
    )
    return_debug: bool = Field(
        False,
        description="If true, return detailed debug info alongside audio.",
    )


class SynthesisResponse(BaseModel):
    audio_base64: str = Field(..., description="Base64-encoded audio data.")
    content_type: str = Field(..., description="MIME type of the audio data.")
    file_path: Optional[str] = Field(
        default=None,
        description="Path on disk where the audio file was written, if applicable.",
    )
    debug: Optional[DebugInfo] = None


class HealthStatus(BaseModel):
    status: str
    emotion_model_loaded: bool
    spacy_model_loaded: bool
    faiss_ready: bool
    tts_configured: bool

