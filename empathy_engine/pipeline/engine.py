from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import List, Tuple
from uuid import uuid4

from empathy_engine.api.schemas import (
    DebugInfo,
    EmotionScore,
    HealthStatus,
    SynthesisRequest,
    SynthesisResponse,
    VocalParamsSchema,
)
from empathy_engine.config.settings import get_settings
from empathy_engine.context.conversation import ConversationContextManager
from empathy_engine.nlp.emotion_correction import correct_emotion_prediction
from empathy_engine.nlp.emotion_model import EmotionPrediction, get_emotion_detector
from empathy_engine.nlp.intensity import calculate_intensity
from empathy_engine.nlp.sarcasm import estimate_sarcasm_score
from empathy_engine.nlp.segmentation import split_sentences
from empathy_engine.utils.logging import get_logger
from empathy_engine.voice.blending import blend_voice_params, select_primary_secondary
from empathy_engine.voice.params import (
    VocalParams,
    get_base_params_for_emotion,
    scale_params,
)
from empathy_engine.voice.ssml import build_sentence_level_ssml, build_simple_ssml
from empathy_engine.voice.tts_client import ElevenLabsClient


logger = get_logger(__name__)


@dataclass
class SynthesisPipeline:
    """High-level orchestration of the EmpathyEngine pipeline.

    This initial version focuses on:
    - emotion detection
    - intensity estimation
    - edge-case handling stubs

    Later iterations will plug in:
    - vocal parameter mapping
    - SSML generation
    - TTS integration
    - vector DB context
    """

    tts_client: ElevenLabsClient | None = field(init=False, default=None)
    context_manager: ConversationContextManager | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._settings = get_settings()
        self._emotion_detector = get_emotion_detector()
        # Context manager / FAISS
        try:
            self.context_manager = ConversationContextManager()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialise conversation context manager: %s", exc)
            self.context_manager = None
        # Lazily initialise TTS client if configured.
        if self._settings.elevenlabs_api_key:
            try:
                self.tts_client = ElevenLabsClient()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to initialise TTS client: %s", exc)
                self.tts_client = None

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResponse:
        text = request.text.strip()
        if not text:
            raise ValueError("Text must not be empty.")

        # Retrieve prior emotional context for this session, if any.
        ctx = (
            self.context_manager.get_recent_emotional_context(request.session_id)
            if self.context_manager is not None
            else None
        )

        sentences = split_sentences(text)
        logger.debug("Split text into %d sentences", len(sentences))

        # Per-sentence emotion detection and intensity.
        sentence_predictions: List[List[EmotionPrediction]] = [
            self._emotion_detector.detect(sent) for sent in sentences
        ]

        sentence_params: List[VocalParams] = []
        all_emotions_flat: List[EmotionPrediction] = []

        for sent, preds in zip(sentences, sentence_predictions):
            # Very short texts: rely more heavily on recent emotional context,
            # or default to a gentle optimistic bias when no context exists.
            words = sent.split()
            is_short = len(words) <= 2

            if is_short and ctx is not None:
                primary = EmotionPrediction(
                    label=ctx.dominant_emotion,
                    score=ctx.intensity,
                )
                secondary = None
            elif is_short and ctx is None:
                primary = EmotionPrediction(label="optimism", score=0.3)
                secondary = None
            else:
                # Apply rule-based corrections to improve accuracy
                corrected_primary = correct_emotion_prediction(sent, preds, min_confidence=0.15)
                if corrected_primary:
                    # Find secondary from remaining predictions
                    remaining = [p for p in preds if p.label.lower() != corrected_primary.label.lower()]
                    _, secondary = select_primary_secondary(remaining)
                    primary = corrected_primary
                else:
                    # Fallback to original selection if correction failed
                    primary, secondary = select_primary_secondary(preds)

            all_emotions_flat.extend(preds)

            primary_conf = primary.score if primary else 0.0
            intensity = calculate_intensity(sent, primary_conf)

            base_primary = get_base_params_for_emotion(primary.label) if primary else get_base_params_for_emotion("neutral")
            scaled_primary = scale_params(base_primary, intensity)

            scaled_secondary: VocalParams | None = None
            if secondary:
                base_secondary = get_base_params_for_emotion(secondary.label)
                scaled_secondary = scale_params(base_secondary, intensity * 0.8)

            blended = blend_voice_params(scaled_primary, scaled_secondary, blend_ratio=0.7)
            sentence_params.append(blended)

        # Global sarcasm heuristic on the full text; slightly dampen intensity/pitch if high.
        sarcasm_score = estimate_sarcasm_score(text, all_emotions_flat)
        if sarcasm_score > 0.5:
            adjusted_params: List[VocalParams] = []
            for p in sentence_params:
                factor = 1.0 - 0.3 * sarcasm_score
                adjusted_params.append(
                    VocalParams(
                        rate=p.rate * factor,
                        pitch=p.pitch * factor,
                        volume_db=p.volume_db,
                        pause_s=p.pause_s,
                        tremor=p.tremor * factor,
                        breathiness=p.breathiness,
                        resonance=p.resonance,
                    )
                )
            sentence_params = adjusted_params

        # Derive an overall "representative" vocal params object for debugging by
        # averaging across sentences.
        overall_params = self._average_vocal_params(sentence_params)
        vocal_params_schema = VocalParamsSchema(
            rate=overall_params.rate,
            pitch=overall_params.pitch,
            volume_db=overall_params.volume_db,
            pause_s=overall_params.pause_s,
            emphasis={},
            tremor=overall_params.tremor,
            breathiness=overall_params.breathiness,
            resonance=overall_params.resonance,
        )

        if len(sentences) > 1:
            ssml = build_sentence_level_ssml(sentences, sentence_params)
        else:
            ssml = build_simple_ssml(text, overall_params)

        # TTS synthesis: if a client is configured, call it; otherwise, return empty audio.
        file_path: str | None = None
        if self.tts_client is not None:
            audio_bytes = await self.tts_client.synthesize(
                text=text,
                ssml=ssml,
                voice_id=request.voice_id,
                audio_format=request.format,
                params=overall_params,
            )
        else:
            audio_bytes = b""

        # Persist audio to disk so it can be inspected / played manually.
        # Files are written under an 'output' directory in the working dir.
        try:
            os.makedirs("output", exist_ok=True)
            ext = request.format
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            file_name = f"empathy_{timestamp}_{uuid4().hex[:8]}.{ext}"
            file_path = os.path.join("output", file_name)
            with open(file_path, "wb") as f:
                f.write(audio_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to write audio file: %s", exc)
            file_path = None

        # Update conversation context / vector store.
        if self.context_manager is not None:
            try:
                self.context_manager.update_context(
                    session_id=request.session_id,
                    text=text,
                    emotions=[e for e in all_emotions_flat],
                    params=overall_params,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to update context: %s", exc)
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        content_type = "audio/mpeg" if request.format == "mp3" else "audio/wav"

        emotions_api: List[EmotionScore] = [
            EmotionScore(label=e.label, score=e.score) for e in all_emotions_flat
        ]

        # Global primary/secondary emotions across the whole text for debug.
        # Apply correction to global emotion as well for consistency
        corrected_global = correct_emotion_prediction(text, all_emotions_flat, min_confidence=0.15)
        if corrected_global:
            remaining_global = [e for e in all_emotions_flat if e.label.lower() != corrected_global.label.lower()]
            global_primary = corrected_global
            _, global_secondary = select_primary_secondary(remaining_global)
        else:
            global_primary, global_secondary = select_primary_secondary(all_emotions_flat)

        debug: DebugInfo | None = None
        if request.return_debug:
            debug_context = None
            if ctx is not None:
                debug_context = {
                    "dominant_emotion": ctx.dominant_emotion,
                    "intensity": ctx.intensity,
                    "history_count": len(ctx.history),
                    "last_text": ctx.history[0].text if ctx.history else None,
                }

            debug = DebugInfo(
                emotions=emotions_api,
                intensity=calculate_intensity(
                    text,
                    global_primary.score if global_primary else 0.0,
                ),
                primary_emotion=global_primary.label if global_primary else "unknown",
                secondary_emotion=global_secondary.label if global_secondary else None,
                vocal_params=vocal_params_schema,
                ssml=ssml,
                context_used=debug_context,
            )

        return SynthesisResponse(
            audio_base64=audio_b64,
            content_type=content_type,
            file_path=file_path,
            debug=debug,
        )

    async def health_check(self) -> HealthStatus:
        # If the imports and lazy initialisers succeeded, the models are considered loaded.
        emotion_model_loaded = True

        # spaCy model is indirectly loaded by the NLP utilities; for now we
        # just assume that if configuration is present we are "ready".
        spacy_model_loaded = True

        # Vector store and TTS will be properly wired in later phases.
        faiss_ready = bool(self.context_manager and self.context_manager.is_ready)
        tts_configured = self.tts_client is not None

        return HealthStatus(
            status="ok",
            emotion_model_loaded=emotion_model_loaded,
            spacy_model_loaded=spacy_model_loaded,
            faiss_ready=faiss_ready,
            tts_configured=tts_configured,
        )

    @staticmethod
    def _average_vocal_params(params_list: List[VocalParams]) -> VocalParams:
        if not params_list:
            return VocalParams(
                rate=1.0,
                pitch=0.0,
                volume_db=0.0,
                pause_s=0.3,
                tremor=0.0,
                breathiness=0.0,
                resonance=0.5,
            )

        n = float(len(params_list))
        total = VocalParams(
            rate=0.0,
            pitch=0.0,
            volume_db=0.0,
            pause_s=0.0,
            tremor=0.0,
            breathiness=0.0,
            resonance=0.0,
        )
        for p in params_list:
            total.rate += p.rate
            total.pitch += p.pitch
            total.volume_db += p.volume_db
            total.pause_s += p.pause_s
            total.tremor += p.tremor
            total.breathiness += p.breathiness
            total.resonance += p.resonance

        return VocalParams(
            rate=total.rate / n,
            pitch=total.pitch / n,
            volume_db=total.volume_db / n,
            pause_s=total.pause_s / n,
            tremor=total.tremor / n,
            breathiness=total.breathiness / n,
            resonance=total.resonance / n,
        )


@lru_cache(maxsize=1)
def get_pipeline() -> SynthesisPipeline:
    """Return a cached SynthesisPipeline instance."""

    return SynthesisPipeline()

