import numpy as np
import pytest

from src.audio import pcm_to_ogg


def test_pcm_to_ogg_returns_bytes():
    audio = np.zeros(24_000, dtype=np.float32)  # 1 s silence
    result = pcm_to_ogg(audio)
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_pcm_to_ogg_ogg_magic_bytes():
    """OGG files start with the 'OggS' capture pattern."""
    audio = np.zeros(24_000, dtype=np.float32)
    result = pcm_to_ogg(audio)
    assert result[:4] == b"OggS"


def test_pcm_to_ogg_short_audio():
    """Very short audio (10 ms) should still produce valid OGG output."""
    audio = np.zeros(240, dtype=np.float32)
    result = pcm_to_ogg(audio)
    assert result[:4] == b"OggS"
