"""/health, CORS, and /v1/tts (OmniVoice faked via the `fake_model` fixture)."""
import io
import os
import sys

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from wordbank.manifest import DONE, Manifest


def _tts_pcm(client, **data):
    r = client.post("/v1/tts", data=data)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "audio/ogg"
    return sf.read(io.BytesIO(r.content))[0]


@pytest.mark.parametrize("bank,model,gemini,status", [
    (True, False, False, 200),
    (False, True, False, 200),     # live-synthesis-only deploy
    (False, False, True, 200),     # Gemini get-or-generate deploy (/v1/speak)
    (False, False, False, 503),    # can serve nothing
])
def test_health_is_healthy_iff_something_can_serve(bank_env, fake_model, monkeypatch,
                                                   bank, model, gemini, status):
    if bank:
        m = Manifest(str(bank_env.db))
        m.add_words(["cat"], "en-us")
        m.set_status(m.iter_rows()[0]["id"], DONE, file_path="en-us/x/0/1_cat.wav")
        m.close()
    if model:
        monkeypatch.setenv("MODEL_ENABLED", "1")
    if gemini:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    from src.server import app
    with TestClient(app) as c:
        r = c.get("/health")
    body = r.json()
    assert r.status_code == status and body["healthy"] is (status == 200)
    assert (body["bank"], body["model_enabled"], body["gemini"]) == (bank, model, gemini)
    assert body["bank_words"] == (1 if bank else 0)


def test_cors_default_allows_any_origin(client):
    """CORS_ORIGINS unset -> wildcard (unchanged local-dev/current behavior)."""
    r = client.get("/health", headers={"Origin": "https://evil.example.com"})
    assert r.headers.get("access-control-allow-origin") == "*"


def test_cors_exposes_speak_metadata_headers(client):
    """Cross-origin JS can only read X-Source/X-Voice if they're exposed."""
    r = client.post("/v1/speak", data={"text": "hello there", "voice": "Puck"},
                    headers={"Origin": "https://app.example.com"})
    assert r.status_code == 200 and r.headers["x-voice"] == "gemini-Puck"
    exposed = {h.strip().lower() for h in r.headers["access-control-expose-headers"].split(",")}
    assert {"x-source", "x-voice"} <= exposed


def test_cors_restricted_rejects_disallowed_origin(bank_env, monkeypatch):
    """CORS_ORIGINS set -> only listed origins are reflected back; the
    middleware is configured at import time from the env, so this needs a
    fresh module reload with the env var already set."""
    import importlib
    import src.server as server_mod
    monkeypatch.setenv("CORS_ORIGINS", " https://app.example.com , ")
    importlib.reload(server_mod)
    try:
        with TestClient(server_mod.app) as c:
            allowed = c.get("/health", headers={"Origin": "https://app.example.com"})
            assert allowed.headers.get("access-control-allow-origin") == "https://app.example.com"
            disallowed = c.get("/health", headers={"Origin": "https://evil.example.com"})
            assert "access-control-allow-origin" not in disallowed.headers
    finally:
        monkeypatch.delenv("CORS_ORIGINS", raising=False)
        importlib.reload(server_mod)  # restore the default wildcard config


def test_tts_text_only_reaches_model_without_reference(client):
    pcm = _tts_pcm(client, text="Hello world")
    assert len(pcm) > 0
    [call] = client.model.calls
    assert call["text"] == "Hello world"
    assert call["ref_audio_path"] is None and call["ref_text"] is None


@pytest.mark.parametrize("ref_text", ["This is my voice.", None])
def test_tts_uploaded_reference_reaches_model_then_is_deleted(client, silent_wav, ref_text):
    data = {"text": "Hello world", **({"ref_text": ref_text} if ref_text else {})}
    r = client.post("/v1/tts", data=data,
                    files={"ref_audio": ("ref.wav", silent_wav, "audio/wav")})
    assert r.status_code == 200, r.text
    [call] = client.model.calls
    assert call["ref_bytes"] == silent_wav          # the upload itself, byte for byte
    assert call["ref_text"] == ref_text             # None -> model auto-transcribes
    assert not os.path.exists(call["ref_audio_path"])   # temp clip cleaned up


def test_tts_seed_and_guidance_determine_the_audio(client):
    """Drives the HTTP contract (request in, audio out): the fake model's output
    is a pure function of its inputs, so a dropped field changes nothing."""
    def synth(seed, guidance="3.5"):
        return _tts_pcm(client, text="hi", instruct="female",
                        guidance_scale=guidance, seed=str(seed))

    a = synth(42)
    assert np.array_equal(a, synth(42))
    assert not np.array_equal(a, synth(43))
    assert not np.array_equal(a, synth(42, guidance="7.0"))


def test_tts_text_too_long(client):
    assert client.post("/v1/tts", data={"text": "x" * 2001}).status_code == 400
    assert client.model.calls == []


def test_tts_ref_audio_too_large(client):
    big = b"\x00" * (10 * 1024 * 1024 + 1)
    r = client.post("/v1/tts", data={"text": "Hello"},
                    files={"ref_audio": ("big.wav", big, "audio/wav")})
    assert r.status_code == 413
    assert client.model.calls == []


def test_tts_disabled_returns_503(bank_env):
    """With MODEL_ENABLED off, /v1/tts is unavailable."""
    from src.server import app
    with TestClient(app) as c:
        assert c.post("/v1/tts", data={"text": "hi"}).status_code == 503


def test_tts_enabled_without_model_build_returns_503(bank_env, monkeypatch):
    """MODEL_ENABLED=1 on a bank-only image (no torch/omnivoice) is a clear
    503, not a 500 from a failed import."""
    monkeypatch.setenv("MODEL_ENABLED", "1")
    monkeypatch.setitem(sys.modules, "src.model", None)    # makes the import fail
    from src.server import app
    with TestClient(app) as c:
        r = c.post("/v1/tts", data={"text": "hi"})
    assert r.status_code == 503 and "not available" in r.json()["detail"]
