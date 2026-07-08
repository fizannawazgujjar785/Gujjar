import os
import asyncio
from unittest import mock
from voice import _speak_async, _play_file


def test_speak_generation_and_cleanup(monkeypatch):
    # Mock edge_tts Communicate.save to write a tiny wav file
    class FakeComm:
        def __init__(self, text, voice):
            self.text = text
            self.voice = voice
        async def save(self, path):
            # create a short silent wav using soundfile
            import soundfile as sf
            import numpy as np
            data = np.zeros((100, 1), dtype='float32')
            sf.write(path, data, 16000, format='WAV')

    monkeypatch.setattr('edge_tts.Communicate', FakeComm)
    # Run generation
    path = asyncio.run(_speak_async("hello"))
    assert os.path.exists(path)
    try:
        _play_file(path)  # should not raise
    finally:
        os.remove(path)
