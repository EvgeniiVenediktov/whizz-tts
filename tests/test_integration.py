"""Live integration: /v1/speak against the real Gemini API (opt-in, paid).

    set -a; . ./.env; set +a
    PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest --integration tests/test_integration.py

One synthesis call (plus Gemini's own empty-response retries, if any). Needs
GEMINI_API_KEY (or GOOGLE_API_KEY), google-genai, and ffmpeg on PATH. The bank
is a tmp one (bank_env); nothing touches the real bank.
"""
import io
import os
import socket

import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from wordbank.manifest import DONE, Manifest

pytestmark = pytest.mark.integration

GEMINI_HOST = "generativelanguage.googleapis.com"
TEXT = "The quick brown fox jumps over the lazy dog."


@pytest.fixture
def live_client(bank_env, monkeypatch):
    """App over a tmp bank with REAL Gemini synthesis; `calls` records the PCM
    each real synthesize_pcm call returned (a spy, so no extra API call)."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        pytest.skip("no GEMINI_API_KEY / GOOGLE_API_KEY in the environment (source .env)")
    pytest.importorskip("google.genai")
    try:
        socket.getaddrinfo(GEMINI_HOST, 443)
    except OSError as exc:
        pytest.fail(f"{GEMINI_HOST} does not resolve ({exc}); fix DNS or run inside "
                    "docker with --dns 8.8.8.8 -- this is not a code failure")
    # bank_env clears the key for the unit suite; put it back for this test.
    monkeypatch.setenv("GEMINI_API_KEY", key)

    import wordbank.gemini_tts as gemini_tts
    real = gemini_tts.synthesize_pcm
    calls = []

    def spy(*args, **kwargs):
        pcm = real(*args, **kwargs)
        calls.append(pcm)
        return pcm

    monkeypatch.setattr(gemini_tts, "synthesize_pcm", spy)
    from src.server import app
    with TestClient(app) as c:
        c.calls, c.paths = calls, bank_env
        yield c


def test_speak_miss_generates_with_real_gemini_then_serves_from_bank(live_client):
    miss = live_client.post("/v1/speak", data={"text": TEXT, "voice": "Puck"})
    assert miss.status_code == 200, miss.text
    assert miss.headers["x-source"] == "gemini" and miss.headers["x-voice"] == "Puck"
    assert miss.content[:4] == b"OggS"

    # The raw payload was unwrapped to bare 16-bit PCM: no RIFF header, no C2PA
    # provenance chunk (gemini-3.8-flash-lite-tts sends a whole WAV file).
    [pcm] = live_client.calls
    assert len(pcm) % 2 == 0 and pcm[:4] != b"RIFF" and b"C2PA" not in pcm

    # A spoken sentence, not silence and not the model reading a prompt aloud.
    audio, sr = sf.read(io.BytesIO(miss.content))
    seconds = len(audio) / sr
    assert 1.0 < seconds < 8.0, f"{seconds:.2f} s for a 9-word sentence"
    assert abs(audio).max() > 0.05, "clip is silent"

    hit = live_client.post("/v1/speak", data={"text": TEXT, "voice": "Puck"})
    assert hit.headers["x-source"] == "bank" and hit.content == miss.content
    assert len(live_client.calls) == 1                     # the hit cost nothing

    m = Manifest(str(live_client.paths.db), table="phrases", text_col="text")
    try:
        [row] = [dict(r) for r in m.iter_rows()]
    finally:
        m.close()
    assert row["status"] == DONE and row["voice"] == "Puck"
