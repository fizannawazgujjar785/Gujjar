"""Async voice I/O utilities for the Gujjar JARVIS assistant.

Improvements over the original:
- Uses Settings from config.get_settings
- Automatic microphone device discovery when MIC_DEVICE_INDEX is None
- Async TTS using edge-tts with streaming support where possible
- Plays audio using sounddevice for better cross-platform performance
- Adds logging, type hints, and exception handling
- Keeps backward-compatible speak/listen synchronous wrappers for simple scripts
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from typing import Optional

import edge_tts
import numpy as np
import soundfile as sf
import sounddevice as sd
import speech_recognition as sr

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def _speak_async(text: str, voice: Optional[str] = None, filepath: Optional[str] = None) -> str:
    """Generate TTS audio file asynchronously and return the path.

    Uses edge-tts Communicate.save to write an mp3 or wav file. Returns
    the path to the generated file.
    """
    voice_name = voice or settings.VOICE_NAME
    if filepath is None:
        fd, filepath = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)

    try:
        communicator = edge_tts.Communicate(text, voice_name)
        # save writes the file asynchronously
        await communicator.save(filepath)
        logger.debug("TTS saved to %s", filepath)
        return filepath
    except Exception as exc:  # pragma: no cover - integration
        logger.exception("TTS generation failed: %s", exc)
        raise


def speak(text: str) -> None:
    """Synchronous convenience wrapper that generates and plays TTS.

    Blocking: will generate TTS and play audio before returning.
    """
    logger.info("Speaking text (len=%d)", len(text))
    try:
        filepath = asyncio.run(_speak_async(text))
        # Read and play using sounddevice for low-latency playback
        data, srate = sf.read(filepath, dtype="float32")
        sd.play(data, srate)
        sd.wait()
    except Exception as exc:
        logger.exception("Error in speak(): %s", exc)
    finally:
        try:
            if "filepath" in locals() and os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            logger.debug("Failed to remove temp tts file")


def _detect_microphone_index(preferred: Optional[int] = None) -> Optional[int]:
    """Try to detect a suitable microphone index if not provided.

    Returns None if the default microphone should be used.
    """
    if preferred is not None:
        return preferred

    try:
        devices = sd.query_devices()
        mic_candidates = [i for i, d in enumerate(devices) if d["max_input_channels"] > 0]
        logger.debug("Detected microphone indices: %s", mic_candidates)
        return mic_candidates[0] if mic_candidates else None
    except Exception:
        logger.exception("Failed to query sound devices")
        return None


def listen(timeout: float = 5.0) -> str:
    """Blocking listen call using SpeechRecognition.

    Adjusts for ambient noise and recognizes via Google's online API.
    Returns empty string on failure.
    """
    recognizer = sr.Recognizer()
    mic_index = _detect_microphone_index(settings.MIC_DEVICE_INDEX)

    try:
        with sr.Microphone(device_index=mic_index) as source:
            logger.info("Listening on mic index %s", mic_index)
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=timeout)
    except Exception as exc:
        logger.exception("Microphone error: %s", exc)
        return ""

    try:
        command = recognizer.recognize_google(audio)
        logger.info("Recognized speech: %s", command)
        return command
    except sr.UnknownValueError:
        logger.info("Speech was unintelligible")
        return ""
    except Exception:
        logger.exception("Speech recognition failed")
        return ""
