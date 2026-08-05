"""Tests for JWT bearer auth (src/auth.py) on the write/synthesis routes.

JWT_SECRET is read at request time (not import time), so these just
setenv/delenv per test -- no client re-creation needed.
"""
import time

import jwt
import pytest

# >=32 bytes: PyJWT warns (InsecureKeyLengthWarning) on shorter HS256 keys.
SECRET = "test-secret-0123456789abcdef0123456789abcdef"


def _token(secret=SECRET, exp_delta=3600, **extra_claims):
    now = int(time.time())
    claims = {"sub": "dev", "iat": now, "exp": now + exp_delta, **extra_claims}
    return jwt.encode(claims, secret, algorithm="HS256")


def test_no_secret_tts_works_unauthenticated(client, monkeypatch):
    monkeypatch.delenv("JWT_SECRET", raising=False)
    r = client.post("/v1/tts", data={"text": "hello"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/ogg"


def test_secret_set_no_header_401(client, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    r = client.post("/v1/tts", data={"text": "hello"})
    assert r.status_code == 401
    assert r.headers.get("www-authenticate") == "Bearer"


def test_secret_set_wrong_secret_401(client, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    bad_token = _token(secret="not-the-secret")
    r = client.post(
        "/v1/tts", data={"text": "hello"},
        headers={"Authorization": f"Bearer {bad_token}"},
    )
    assert r.status_code == 401


def test_secret_set_expired_token_401(client, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    expired = _token(exp_delta=-10)
    r = client.post(
        "/v1/tts", data={"text": "hello"},
        headers={"Authorization": f"Bearer {expired}"},
    )
    assert r.status_code == 401


def test_secret_set_valid_token_200(client, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    token = _token()
    r = client.post(
        "/v1/tts", data={"text": "hello"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/ogg"


def test_gemini_tts_requires_auth(client, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    r = client.post("/v1/gemini/tts", data={"text": "hello"})
    assert r.status_code == 401


def test_save_voice_requires_auth(client, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    r = client.post(
        "/v1/voices",
        data={"name": "x", "lang": "en-us", "ref_text": "y"},
        files={"audio": ("x.wav", b"RIFF....WAVEfmt ", "audio/wav")},
    )
    assert r.status_code == 401


def test_public_routes_unaffected_by_secret(client, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    assert client.get("/health").status_code != 401
    assert client.get("/v1/voices").status_code != 401
