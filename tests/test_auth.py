"""JWT bearer auth (src/auth.py) on the synthesis routes.

JWT_SECRET is read at request time (not import time), so these just
setenv/delenv per test -- no client re-creation needed. With no secret the
routes are open; every other server test exercises that.
"""
import time

import jwt
import pytest

# >=32 bytes: PyJWT warns (InsecureKeyLengthWarning) on shorter HS256 keys.
SECRET = "test-secret-0123456789abcdef0123456789abcdef"

PROTECTED = [
    ("/v1/speak", {"text": "hello"}),
    ("/v1/gemini/tts", {"text": "hello"}),
]


def _token(secret=SECRET, exp_delta=3600, **extra_claims):
    now = int(time.time())
    claims = {"sub": "dev", "iat": now, "exp": now + exp_delta, **extra_claims}
    return jwt.encode(claims, secret, algorithm="HS256")


def _post(client, route, data, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.post(route, data=data, headers=headers)


@pytest.mark.parametrize("route,data", PROTECTED)
def test_protected_route_needs_valid_token_when_secret_set(client, monkeypatch, route, data):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    missing = _post(client, route, data)
    assert missing.status_code == 401 and missing.headers.get("www-authenticate") == "Bearer"
    assert _post(client, route, data, _token(secret="wrong-secret-0123456789abcdef0123456789")).status_code == 401
    assert _post(client, route, data, _token(exp_delta=-10)).status_code == 401
    no_exp = jwt.encode({"sub": "dev"}, SECRET, algorithm="HS256")     # would never expire
    assert _post(client, route, data, no_exp).status_code == 401
    # nothing was synthesized or written by the rejected calls
    assert client.gemini.calls == []
    assert not client.paths.db.exists()

    assert _post(client, route, data, _token()).status_code == 200


def test_audience_and_issuer_are_enforced_when_configured(client, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    monkeypatch.setenv("JWT_AUDIENCE", "tts")
    monkeypatch.setenv("JWT_ISSUER", "whizz")
    route, data = PROTECTED[0]
    assert _post(client, route, data, _token(aud="other", iss="whizz")).status_code == 401
    assert _post(client, route, data, _token(aud="tts", iss="other")).status_code == 401
    assert _post(client, route, data, _token(aud="tts", iss="whizz")).status_code == 200


def test_public_routes_unaffected_by_secret(client, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    for path in ("/health", "/v1/gemini/voices", "/v1/languages"):
        assert client.get(path).status_code == 200, path
    assert client.get("/v1/search", params={"prefix": "a"}).status_code == 200


def test_oversized_body_is_413_before_anything_parses_it(client, monkeypatch):
    """FastAPI parses a form before require_auth runs, so the body cap comes
    first: by Content-Length, and by counting a chunked body."""
    monkeypatch.setenv("JWT_SECRET", SECRET)
    big = b"text=" + b"x" * (64 * 1024)
    form = {"Content-Type": "application/x-www-form-urlencoded"}
    assert client.post("/v1/speak", content=big, headers=form).status_code == 413
    chunked = client.post("/v1/speak", content=iter([big[:40_000], big[40_000:]]), headers=form)
    assert chunked.status_code == 413
    assert _post(client, "/v1/speak", {"text": "hello"}).status_code == 401
