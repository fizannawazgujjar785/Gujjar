import os
import asyncio
import speech_recognition as sr
import edge_tts
import pygame

from config import VOICE_NAME, MIC_DEVICE_INDEX

pygame.mixer.init()

async def _speak_async(text):
    communicate = edge_tts.Communicate(text, VOICE_NAME)
    await communicate.save("voice.mp3")


def speak(text):
    print("Jarvis:", text)
    try:
        asyncio.run(_speak_async(text))
        pygame.mixer.music.load("voice.mp3")
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.music.unload()
    except Exception as exc:
        print("Voice error:", exc)
    finally:
        if os.path.exists("voice.mp3"):
            os.remove("voice.mp3")


def listen():
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone(device_index=MIC_DEVICE_INDEX) as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source)
    except Exception as exc:
        print("Microphone error:", exc)
        return ""

    try:
        command = recognizer.recognize_google(audio)
        print("You:", command)
        return command
    except Exception:
        return ""