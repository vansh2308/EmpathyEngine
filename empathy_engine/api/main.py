from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from empathy_engine.api.schemas import (
    HealthStatus,
    SynthesisRequest,
    SynthesisResponse,
)
from empathy_engine.config.settings import get_settings
from empathy_engine.pipeline.engine import get_pipeline, SynthesisPipeline


def create_app() -> FastAPI:
    app = FastAPI(
        title="EmpathyEngine",
        description="Emotion-aware TTS service that modulates vocal parameters based on detected emotions.",
        version="0.1.0",
    )

    @app.on_event("startup")
    async def on_startup() -> None:
        # Trigger lazy initialization of settings and pipeline at startup.
        get_settings()
        get_pipeline()

    def get_synthesis_pipeline() -> SynthesisPipeline:
        return get_pipeline()

    @app.post("/synthesize", response_model=SynthesisResponse)
    async def synthesize(
        request: SynthesisRequest,
        pipeline: SynthesisPipeline = Depends(get_synthesis_pipeline),
    ) -> SynthesisResponse:
        try:
            return await pipeline.synthesize(request)
        except Exception as exc:  # noqa: BLE001
            # In a real system, use structured logging and avoid exposing internals.
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/health", response_model=HealthStatus)
    async def health(
        pipeline: SynthesisPipeline = Depends(get_synthesis_pipeline),
    ) -> HealthStatus:
        status = await pipeline.health_check()
        return status

    return app


app = create_app()

