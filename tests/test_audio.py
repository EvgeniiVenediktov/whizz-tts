"""src/audio.py: PCM -> OGG/Opus via ffmpeg."""
import io

import soundfile as sf

from src.audio import pcm16_to_ogg
from tests.conftest import noise_pcm16


def test_pcm16_to_ogg_round_trips_the_whole_clip():
    ogg = pcm16_to_ogg(noise_pcm16(1.0))
    pcm, sr = sf.read(io.BytesIO(ogg))
    assert ogg[:4] == b"OggS"
    assert abs(len(pcm) / sr - 1.0) < 0.05


def test_bitrate_reaches_the_encoder():
    pcm = noise_pcm16(2.0)
    bank = pcm16_to_ogg(pcm, bitrate="16k", application="voip")
    live = pcm16_to_ogg(pcm, bitrate="64k")
    assert len(bank) * 8 / 2 < 24_000 < len(live) * 8 / 2
