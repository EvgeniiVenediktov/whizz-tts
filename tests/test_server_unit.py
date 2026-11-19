"""/health and CORS."""
import pytest
from fastapi.testclient import TestClient

from wordbank.manifest import DONE, Manifest


@pytest.mark.parametrize("bank,gemini,status", [
    (True, False, 200),
    (False, True, 200),     # Gemini get-or-generate deploy (/v1/speak)
    (False, False, 503),    # can serve nothing
])
def test_health_is_healthy_iff_something_can_serve(bank_env, monkeypatch, bank, gemini, status):
    if bank:
        m = Manifest(str(bank_env.db))
        m.add_words(["cat"], "en-us")
        m.set_status(m.iter_rows()[0]["id"], DONE, file_path="en-us/x/0/1_cat.wav")
        m.close()
    if gemini:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    from src.server import app
    with TestClient(app) as c:
        r = c.get("/health")
    body = r.json()
    assert r.status_code == status and body["healthy"] is (status == 200)
    assert (body["bank"], body["gemini"]) == (bank, gemini)
    assert body["bank_words"] == (1 if bank else 0)


def test_cors_default_allows_any_origin(client):
    """CORS_ORIGINS unset -> wildcard (unchanged local-dev/current behavior)."""
    r = client.get("/health", headers={"Origin": "https://evil.example.com"})
    assert r.headers.get("access-control-allow-origin") == "*"


def test_cors_exposes_speak_metadata_headers(client):
    """Cross-origin JS can only read X-Source/X-Voice if they're exposed."""
    r = client.post("/v1/speak", data={"text": "hello there", "voice": "Puck"},
                    headers={"Origin": "https://app.example.com"})
    assert r.status_code == 200 and r.headers["x-voice"] == "Puck"
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


