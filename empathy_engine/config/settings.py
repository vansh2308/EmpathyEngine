from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_prefix="EMPATHY_", env_file=".env", extra="ignore")

    # General
    env: str = "development"
    log_level: str = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # NLP models
    emotion_model_name: str = "cardiffnlp/twitter-roberta-base-emotion"
    spacy_model_name: str = "en_core_web_sm"
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # TTS / ElevenLabs
    elevenlabs_api_key: Optional[str] = None
    elevenlabs_base_url: str = "https://api.elevenlabs.io"
    elevenlabs_default_voice_id: str = "QZfuYwrJM2qQevLoK38m"
    elevenlabs_model_id: str = "eleven_multilingual_v2"
    tts_timeout_seconds: float = 30.0

    # Vector DB / FAISS
    faiss_index_path: str = "data/faiss_index.bin"
    faiss_metadata_path: str = "data/faiss_metadata.json"
    embedding_dimension: int = 384
    faiss_use_inner_product: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""

    return Settings()


settings = get_settings()

