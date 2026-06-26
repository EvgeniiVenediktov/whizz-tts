"""Unit tests for the /v1/tts and /health endpoints (model mocked)."""
import io

import pytest


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "device" in body
    assert "model" in body


def test_tts_no_ref(client):
    """Text only — auto voice mode."""
    r = client.post("/v1/tts", data={"text": "Hello world"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/ogg"
    assert r.content[:4] == b"OggS"


def test_tts_with_ref_audio_and_text(client, silent_wav):
    r = client.post(
        "/v1/tts",
        data={"text": "Hello world", "ref_text": "This is my voice."},
        files={"ref_audio": ("ref.wav", silent_wav, "audio/wav")},
    )
    assert r.status_code == 200
    assert r.content[:4] == b"OggS"


def test_tts_with_ref_audio_no_ref_text(client, silent_wav):
    """ref_audio without ref_text — model should auto-transcribe."""
    r = client.post(
        "/v1/tts",
        data={"text": "Hello world"},
        files={"ref_audio": ("ref.wav", silent_wav, "audio/wav")},
    )
    assert r.status_code == 200
    assert r.content[:4] == b"OggS"


def test_tts_text_too_long(client):
    r = client.post("/v1/tts", data={"text": "x" * 2001})
    assert r.status_code == 400


def test_tts_ref_audio_too_large(client):
    big = b"\x00" * (10 * 1024 * 1024 + 1)
    r = client.post(
        "/v1/tts",
        data={"text": "Hello"},
        files={"ref_audio": ("big.wav", big, "audio/wav")},
    )
    assert r.status_code == 413


def test_tts_missing_text(client):
    r = client.post("/v1/tts", data={})
    assert r.status_code == 422


def test_tts_forwards_guidance_and_seed(client):
    """guidance_scale (float) and seed (int) are parsed and forwarded to generate()."""
    import numpy as np
    import src.model as model_mod
    from unittest.mock import patch

    captured = {}

    def capture(text, ref_audio_path=None, ref_text=None,
                instruct=None, guidance_scale=None, seed=None):
        captured.update(instruct=instruct, guidance_scale=guidance_scale, seed=seed)
        return np.zeros(1000, dtype=np.float32)

    with patch.object(model_mod, "generate", capture):
        r = client.post(
            "/v1/tts",
            data={"text": "hi", "instruct": "female",
                  "guidance_scale": "3.5", "seed": "42"},
        )
    assert r.status_code == 200
    assert captured == {"instruct": "female", "guidance_scale": 3.5, "seed": 42}


def test_tts_disabled_returns_503(monkeypatch, tmp_path):
    """With MODEL_ENABLED off, /v1/tts is unavailable but the bank still serves."""
    monkeypatch.setenv("MODEL_ENABLED", "0")
    monkeypatch.setenv("BANK_DB", str(tmp_path / "none.db"))
    import src.server as server_mod
    from fastapi.testclient import TestClient
    with TestClient(server_mod.app) as c:
        assert c.post("/v1/tts", data={"text": "hi"}).status_code == 503
        assert c.get("/health").json()["model_enabled"] is False
