from __future__ import annotations

import logging
from typing import Optional

from empathy_engine.config.settings import get_settings


def configure_logging(level: Optional[str] = None) -> None:
    """Configure root logging for the application."""

    settings = get_settings()
    log_level = (level or settings.log_level).upper()

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger instance with the given name."""

    return logging.getLogger(name)

