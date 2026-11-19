"""src/audio.py: PCM -> OGG/Opus via ffmpeg."""
import io
import os
import pathlib
import subprocess
import sys

import soundfile as sf

from src.audio import pcm16_to_ogg
from tests.conftest import noise_pcm16
from wordbank.gemini_tts import pcm_to_wav_bytes

REPO_ROOT = pathlib.Path(__file__).parent.parent


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


def test_encode_wavs_re_encodes_a_copy_older_than_its_wav(tmp_path):
    """A re-take rewrites the WAV: its stale OGG must not survive, while an
    up-to-date copy is left alone."""
    wav = tmp_path / "words" / "en" / "1_cat.wav"
    ogg = tmp_path / "ogg" / "en" / "1_cat.ogg"
    for f in (wav, ogg):
        f.parent.mkdir(parents=True)
    wav.write_bytes(pcm_to_wav_bytes(noise_pcm16(0.5)))
    ogg.write_bytes(b"stale")
    os.utime(ogg, (wav.stat().st_mtime - 10,) * 2)

    def encode():
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "encode_wavs.py"),
                        str(tmp_path / "words"), str(tmp_path / "ogg")], check=True,
                       capture_output=True)
    encode()
    assert ogg.read_bytes()[:4] == b"OggS"
    encoded = ogg.stat().st_mtime_ns
    encode()
    assert ogg.stat().st_mtime_ns == encoded
