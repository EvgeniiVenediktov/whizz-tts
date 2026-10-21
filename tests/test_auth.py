"""JWT bearer auth (src/auth.py) on the write/synthesis routes.

JWT_SECRET is read at request time (not import time), so these just
setenv/delenv per test -- no client re-creation needed.
"""
import time

import jwt
import pytest

from tests.conftest import make_wav_bytes

# >=32 bytes: PyJWT warns (InsecureKeyLengthWarning) on shorter HS256 keys.
SECRET = "test-secret-0123456789abcdef0123456789abcdef"

PROTECTED = [
    ("/v1/tts", {"text": "hello"}, None),
    ("/v1/speak", {"text": "hello"}, None),
    ("/v1/gemini/tts", {"text": "hello"}, None),
    ("/v1/voices", {"name": "x", "lang": "en-us", "ref_text": "y"},
     {"audio": ("x.wav", make_wav_bytes(0.1), "audio/wav")}),
]


def _token(secret=SECRET, exp_delta=3600, **extra_claims):
    now = int(time.time())
    claims = {"sub": "dev", "iat": now, "exp": now + exp_delta, **extra_claims}
    return jwt.encode(claims, secret, algorithm="HS256")


def _post(client, route, data, files, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.post(route, data=data, files=files, headers=headers)


@pytest.mark.parametrize("route,data,files", PROTECTED)
def test_protected_route_open_without_secret(client, route, data, files):
    assert _post(client, route, data, files).status_code == 200


@pytest.mark.parametrize("route,data,files", PROTECTED)
def test_protected_route_needs_valid_token_when_secret_set(client, monkeypatch, route, data, files):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    missing = _post(client, route, data, files)
    assert missing.status_code == 401 and missing.headers.get("www-authenticate") == "Bearer"
    assert _post(client, route, data, files, _token(secret="wrong-secret-0123456789abcdef0123456789")).status_code == 401
    assert _post(client, route, data, files, _token(exp_delta=-10)).status_code == 401
    # nothing was synthesized or written by the rejected calls
    assert client.model.calls == [] and client.gemini.calls == []
    assert client.get("/v1/voices").json()["voices"] == []

    assert _post(client, route, data, files, _token()).status_code == 200


def test_audience_and_issuer_are_enforced_when_configured(client, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    monkeypatch.setenv("JWT_AUDIENCE", "tts")
    monkeypatch.setenv("JWT_ISSUER", "whizz")
    route, data, files = PROTECTED[0]
    assert _post(client, route, data, files, _token(aud="other", iss="whizz")).status_code == 401
    assert _post(client, route, data, files, _token(aud="tts", iss="other")).status_code == 401
    assert _post(client, route, data, files, _token(aud="tts", iss="whizz")).status_code == 200


def test_public_routes_unaffected_by_secret(client, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    for path in ("/health", "/v1/voices", "/v1/gemini/voices", "/v1/languages"):
        assert client.get(path).status_code == 200, path
    assert client.get("/v1/search", params={"prefix": "a"}).status_code == 200
