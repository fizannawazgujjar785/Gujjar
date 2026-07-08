"""Async voice I/O utilities for the Gujjar JARVIS assistant.

Improvements over the original:
- Uses Settings from config.get_settings
- Automatic microphone device discovery when MIC_DEVICE_INDEX is None
- Async TTS using edge-tts with streaming support where possible
- Plays audio using sounddevice for better cross-platform performance
- Falls back to pydub->ffmpeg conversion or pygame playback for mp3 compatibility
- Adds logging, type hints, and exception handling
- Keeps backward-compatible speak/listen synchronous wrappers for simple scripts

Notes:
- Requires ffmpeg on PATH if pydub is used to convert mp3 -> wav.
- soundfile/libsnfile may not support mp3 on all platforms. The code handles
  that by attempting conversion or using pygame as a last resort.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from typing import Optional

import edge_tts
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Optional dependencies
try:
    from pydub import AudioSegment  # type: ignore
except Exception:
    AudioSegment = None  # type: ignore

try:
    import pygame  # type: ignore
    _pygame_available = True
except Exception:
    pygame = None  # type: ignore
    _pygame_available = False


async def _speak_async(text: str, voice: Optional[str] = None, filepath: Optional[str] = None) -> str:
    """Generate TTS audio file asynchronously and return the path.

    Uses edge-tts Communicate.save to write an mp3 file by default. Returns
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


def _play_file(filepath: str) -> None:
    """Play an audio file. Attempts to use soundfile+sounddevice for WAV/FLAC,
    falls back to converting mp3->wav via pydub if available, else uses pygame
    to play mp3.
    """
    try:
        # Try reading with soundfile (supports WAV/FLAC and others if libsndfile built with support)
        data, srate = sf.read(filepath, dtype="float32")
        sd.play(data, srate)
        sd.wait()
        return
    except RuntimeError as exc:
        logger.debug("soundfile failed to read %s: %s", filepath, exc)
    except Exception as exc:
        logger.exception("Error playing via sounddevice: %s", exc)

    # If we reached here the format was likely mp3 and soundfile cannot read it.
    if AudioSegment:
        try:
            logger.debug("Converting mp3 to wav using pydub: %s", filepath)
            audio = AudioSegment.from_file(filepath)
            fd, wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            audio.export(wav_path, format="wav")
            data, srate = sf.read(wav_path, dtype="float32")
            sd.play(data, srate)
            sd.wait()
            try:
                os.remove(wav_path)
            except Exception:
                logger.debug("Failed to remove temporary wav file %s", wav_path)
            return
        except Exception as exc:  # pragma: no cover - conversion depends on ffmpeg
            logger.exception("pydub conversion failed: %s", exc)

    # Final fallback: pygame mixer (may support mp3 depending on system codecs)
    if _pygame_available and pygame:
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.music.unload()
            return
        except Exception as exc:  # pragma: no cover - platform codec dependent
            logger.exception("pygame fallback playback failed: %s", exc)

    logger.error("Unable to play audio file %s - no compatible player available", filepath)


def speak(text: str) -> None:
    """Synchronous convenience wrapper that generates and plays TTS.

    Blocking: will generate TTS and play audio before returning.
    """
    logger.info("Speaking text (len=%d)", len(text))
    filepath = None
    try:
        filepath = asyncio.run(_speak_async(text))
        _play_file(filepath)
    except Exception as exc:
        logger.exception("Error in speak(): %s", exc)
    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                logger.debug("Failed to remove temp tts file %s", filepath)


def _detect_microphone_index(preferred: Optional[int] = None) -> Optional[int]:
    """Detect microphone index compatible with SpeechRecognition (PyAudio).

    SpeechRecognition exposes a list of microphone names via
    sr.Microphone.list_microphone_names(); their indices correspond to the
    device_index accepted by sr.Microphone. We prefer that over sounddevice
    enumeration to ensure compatibility with PyAudio.
    """
    if preferred is not None:
        logger.debug("Using preferred MIC_DEVICE_INDEX=%s", preferred)
        return preferred

    try:
        names = sr.Microphone.list_microphone_names()
        logger.debug("Available microphones: %s", names)
        if not names:
            return None

        # Try to pick a non-empty and non-virtual microphone name
        for idx, name in enumerate(names):
            if name and "stereo mix" not in name.lower():
                logger.debug("Selected microphone index %d (%s)", idx, name)
                return idx

        # Fallback to first index
        logger.debug("Falling back to microphone index 0")
        return 0
    except Exception:
        logger.exception("Failed to list microphones via SpeechRecognition")
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
    except sr.WaitTimeoutError:
        logger.info("Listening timed out after %s seconds", timeout)
        return ""
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
