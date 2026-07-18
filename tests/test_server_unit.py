"""Unit tests for the /v1/tts and /health endpoints (model mocked)."""


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "device" in body
    assert "model" in body
    # bank available (real wordbank.db) -> healthy regardless of MODEL_ENABLED.
    assert body["healthy"] is True
    assert body["bank"] is True


def test_cors_default_allows_any_origin(client):
    """CORS_ORIGINS unset -> wildcard (unchanged local-dev/current behavior)."""
    r = client.get("/health", headers={"Origin": "https://evil.example.com"})
    assert r.headers.get("access-control-allow-origin") == "*"


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
    """seed (int) determines the generated audio: same seed -> identical
    response bytes, different seed -> different bytes.

    This drives the real HTTP contract (request in, audio out) rather than
    inspecting the kwargs generate() was called with, so it survives a
    refactor to how guidance_scale/seed are threaded through internally
    (e.g. bundled into a dataclass) as long as the seed still determines
    the output.
    """
    import io

    import numpy as np
    import soundfile as sf
    import src.model as model_mod
    from unittest.mock import patch

    import zlib

    def deterministic_generate(text, ref_audio_path=None, ref_text=None,
                                instruct=None, guidance_scale=None, seed=None):
        # Stand-in model: PCM is a pure function of every value the handler
        # forwards, so dropping any one of them changes the audio.
        key = f"{seed}|{guidance_scale}|{instruct}|{text}".encode()
        rng = np.random.RandomState(zlib.crc32(key))
        return rng.uniform(-1, 1, 4800).astype(np.float32)

    def synth(seed, guidance="3.5"):
        with patch.object(model_mod, "generate", deterministic_generate):
            r = client.post(
                "/v1/tts",
                data={"text": "hi", "instruct": "female",
                      "guidance_scale": guidance, "seed": str(seed)},
            )
        assert r.status_code == 200
        assert r.headers["content-type"] == "audio/ogg"
        pcm, _ = sf.read(io.BytesIO(r.content))
        return pcm

    pcm_seed_42_a = synth(42)
    pcm_seed_42_b = synth(42)
    pcm_seed_43 = synth(43)

    assert np.array_equal(pcm_seed_42_a, pcm_seed_42_b)        # same seed -> identical audio
    assert not np.array_equal(pcm_seed_42_a, pcm_seed_43)      # different seed -> different audio

    # guidance_scale must reach the model too (the endpoint accepts it, so a
    # handler that silently dropped it would otherwise go unnoticed).
    assert not np.array_equal(pcm_seed_42_a, synth(42, guidance="7.0"))


def test_tts_disabled_returns_503(monkeypatch, tmp_path):
    """With MODEL_ENABLED off, /v1/tts is unavailable but the bank still serves."""
    monkeypatch.setenv("MODEL_ENABLED", "0")
    monkeypatch.setenv("BANK_DB", str(tmp_path / "none.db"))
    import src.server as server_mod
    from fastapi.testclient import TestClient
    with TestClient(server_mod.app) as c:
        assert c.post("/v1/tts", data={"text": "hi"}).status_code == 503
        assert c.get("/health").json()["model_enabled"] is False


# NOTE: the tests below each spin up their own TestClient (re-running lifespan
# on the shared module-level app, like test_tts_disabled_returns_503 above) or
# reload the module outright, both of which mutate/replace state the module-
# scoped `client` fixture depends on. They must stay after every `client`-based
# test in this file.

def test_health_503_when_bank_unavailable_and_model_disabled(monkeypatch, tmp_path):
    """Neither the bank nor live synthesis can serve anything -> 503."""
    monkeypatch.setenv("MODEL_ENABLED", "0")
    monkeypatch.setenv("BANK_DB", str(tmp_path / "none.db"))
    import src.server as server_mod
    from fastapi.testclient import TestClient
    with TestClient(server_mod.app) as c:
        r = c.get("/health")
        assert r.status_code == 503
        body = r.json()
        assert body["healthy"] is False
        assert body["bank"] is False
        assert body["model_enabled"] is False
        # body shape unchanged -- same keys as the healthy case.
        assert set(body) >= {"status", "healthy", "model", "model_enabled",
                              "bank", "bank_path", "bank_words"}


def test_health_200_when_model_only(monkeypatch, tmp_path):
    """No bank, but MODEL_ENABLED on -> still healthy (model-only deploy is
    legitimate; this is the regression test for the bank-only 503 mistake)."""
    monkeypatch.setenv("MODEL_ENABLED", "1")
    monkeypatch.setenv("BANK_DB", str(tmp_path / "none.db"))
    import src.server as server_mod
    from fastapi.testclient import TestClient
    with TestClient(server_mod.app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["healthy"] is True
        assert body["bank"] is False
        assert body["model_enabled"] is True


def test_cors_restricted_rejects_disallowed_origin(monkeypatch):
    """CORS_ORIGINS set -> only listed origins are reflected back; the
    middleware is configured at import time from the env, so this needs a
    fresh module reload with the env var already set."""
    import importlib
    import src.server as server_mod
    monkeypatch.setenv("CORS_ORIGINS", " https://app.example.com , ")
    importlib.reload(server_mod)
    try:
        from fastapi.testclient import TestClient
        with TestClient(server_mod.app) as c:
            allowed = c.get("/health", headers={"Origin": "https://app.example.com"})
            assert allowed.headers.get("access-control-allow-origin") == "https://app.example.com"

            disallowed = c.get("/health", headers={"Origin": "https://evil.example.com"})
            assert "access-control-allow-origin" not in disallowed.headers
    finally:
        monkeypatch.delenv("CORS_ORIGINS", raising=False)
        importlib.reload(server_mod)  # restore the default wildcard config
