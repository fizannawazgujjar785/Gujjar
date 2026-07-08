"""Configuration loader for Gujjar JARVIS assistant.

Loads environment variables from a .env file and exposes a Settings dataclass.
Reads API keys only from environment and never hardcodes secrets.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment.

    All secrets must be provided via environment variables or a .env file.
    """

    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    SEARCH_API_KEY: Optional[str] = os.getenv("SEARCH_API_KEY")

    WAKE_WORD: str = os.getenv("WAKE_WORD", "jarvis")
    ASSISTANT_NAME: str = os.getenv("ASSISTANT_NAME", "Jarvis")
    USER_NAME: str = os.getenv("USER_NAME", "Fizan")

    VOICE_NAME: str = os.getenv("VOICE_NAME", "en-US-GuyNeural")
    MIC_DEVICE_INDEX: Optional[int] = (
        int(os.getenv("MIC_DEVICE_INDEX")) if os.getenv("MIC_DEVICE_INDEX") else None
    )


def get_settings() -> Settings:
    """Return a Settings instance and log redacted secrets.

    Secrets will never be logged in full; presence is logged only.
    """
    s = Settings()
    logger.debug("Loaded settings: OPENAI_API_KEY=%s, GEMINI_API_KEY=%s", 
                 bool(s.OPENAI_API_KEY), bool(s.GEMINI_API_KEY))
    return s
