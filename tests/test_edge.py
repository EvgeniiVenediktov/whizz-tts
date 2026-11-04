"""Edge mode: /v1/speak serves its own bank and fetches misses from an upstream
whizz-tts server (SPEAK_UPSTREAM_URL) instead of calling Gemini.

Most tests point the edge at a stdlib HTTP server standing in for upstream, so
the real urllib client is exercised; the last one runs a real upstream server
in a subprocess (Gemini faked there) to pin the edge <-> upstream contract.
"""
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import jwt
import pytest
from fastapi.testclient import TestClient

from wordbank.manifest import DONE, Manifest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN = "edge-token"


class FakeUpstream:
    """Answers POST /v1/speak like the real server would; `respond(fields)`
    -> (status, headers, body) can be swapped per test. Records every request."""

    def __init__(self):
        self.requests = []
        self.delay = 0.0
        self.respond = self._ok
        up = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                fields = dict(urllib.parse.parse_qsl(raw.decode()))
                up.requests.append({"path": self.path,
                                    "auth": self.headers.get("Authorization"), **fields})
                time.sleep(up.delay)
                status, headers, body = up.respond(fields)
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @staticmethod
    def _ok(fields):
        voice = f"gemini-{fields.get('voice', 'Kore')}"
        body = b"OggS" + f"|{fields['text']}|{voice}|{fields.get('instruct', '')}".encode()
        return 200, {"Content-Type": "audio/ogg", "X-Source": "gemini", "X-Voice": voice}, body


@pytest.fixture
def upstream():
    up = FakeUpstream()
    yield up
    up.server.shutdown()


@pytest.fixture
def edge(bank_env, fake_gemini, upstream, monkeypatch):
    monkeypatch.setenv("SPEAK_UPSTREAM_URL", upstream.url + "/")   # trailing slash tolerated
    monkeypatch.setenv("SPEAK_UPSTREAM_TOKEN", TOKEN)
    from src.server import app
    with TestClient(app) as c:
        c.upstream, c.gemini, c.paths = upstream, fake_gemini, bank_env
        yield c


def _speak(client, **data):
    return client.post("/v1/speak", data=data)


def _phrases(paths):
    m = Manifest(str(paths.db), table="phrases", text_col="text")
    try:
        return [dict(r) for r in m.iter_rows()]
    finally:
        m.close()


def test_miss_is_fetched_from_upstream_saved_and_then_served_locally(edge):
    r = _speak(edge, text="hello there", voice="puck")
    assert r.status_code == 200, r.text
    assert r.headers["x-source"] == "upstream" and r.headers["x-voice"] == "gemini-Puck"
    assert r.content == b"OggS|hello there|gemini-Puck|"

    [req] = edge.upstream.requests
    assert req == {"path": "/v1/speak", "auth": f"Bearer {TOKEN}",
                   "text": "hello there", "lang": "en-us", "voice": "Puck"}
    [row] = _phrases(edge.paths)
    assert row["voice"] == "gemini-Puck" and row["status"] == DONE
    assert (edge.paths.ogg / (row["file_path"][:-4] + ".ogg")).read_bytes() == r.content

    again = _speak(edge, text="Hello there", voice="Puck")
    assert again.headers["x-source"] == "bank" and again.content == r.content
    assert len(edge.upstream.requests) == 1
    assert edge.gemini.calls == []                  # an edge never calls Gemini itself


def test_packaged_bank_serves_without_asking_upstream(edge):
    rel = "en-us/gemini-Kore/0/1_cat.wav"
    ogg = edge.paths.ogg / "en-us/gemini-Kore/0/1_cat.ogg"
    ogg.parent.mkdir(parents=True)
    ogg.write_bytes(b"OggS-cat")
    m = Manifest(str(edge.paths.db))
    m.add_words(["cat"], "en-us", voice="Kore")
    m.set_status(m.iter_rows()[0]["id"], DONE, file_path=rel, voice="Kore")
    m.close()

    r = _speak(edge, text="cat")
    assert r.headers["x-source"] == "bank" and r.content == b"OggS-cat"
    assert edge.upstream.requests == []


def test_no_voice_is_forwarded_as_none_and_filed_under_the_voice_upstream_used(edge):
    """Upstream applies its own default/any-banked rule; a word-bank hit there
    reports the bare name, filed here as the phrase voice gemini-<Name>."""
    edge.upstream.respond = lambda f: (200, {"X-Source": "bank", "X-Voice": "Kore"}, b"OggS-word")
    r = _speak(edge, text="cat")
    assert "voice" not in edge.upstream.requests[0]
    assert r.headers["x-voice"] == "gemini-Kore"
    assert [p["voice"] for p in _phrases(edge.paths)] == ["gemini-Kore"]

    pinned = _speak(edge, text="cat", voice="Kore")   # now a local hit in that voice
    assert pinned.headers["x-source"] == "bank" and len(edge.upstream.requests) == 1


def test_instruct_is_forwarded_and_never_saved(edge):
    for _ in range(2):
        r = _speak(edge, text="hello there", voice="Puck", instruct="whisper")
        assert r.headers["x-source"] == "upstream" and r.content.endswith(b"|whisper")
    assert [q["instruct"] for q in edge.upstream.requests] == ["whisper", "whisper"]
    assert _phrases(edge.paths) == []


@pytest.mark.parametrize("status,body,want_status,want_detail", [
    (400, b'{"detail":"unknown Gemini voice"}', 400, "unknown Gemini voice"),
    (401, b'{"detail":"invalid token"}', 502, "rejected the edge's token"),
    (502, b'{"detail":"Gemini TTS failed: quota"}', 502, "Gemini TTS failed: quota"),
    (200, b'{"not":"audio"}', 502, "not OGG audio"),
])
def test_upstream_failures_map_to_clear_errors_and_save_nothing(edge, status, body,
                                                               want_status, want_detail):
    edge.upstream.respond = lambda f: (status, {"Content-Type": "application/json"}, body)
    r = _speak(edge, text="hello there", voice="Puck")
    assert r.status_code == want_status and want_detail in r.json()["detail"]
    assert _phrases(edge.paths) == []


def test_unreachable_upstream_is_502(edge, monkeypatch):
    with socket.socket() as s:            # a port nothing listens on
        s.bind(("127.0.0.1", 0))
        dead = f"http://127.0.0.1:{s.getsockname()[1]}"
    monkeypatch.setenv("SPEAK_UPSTREAM_URL", dead)
    r = _speak(edge, text="hello there")
    assert r.status_code == 502 and "unreachable" in r.json()["detail"]


def test_slow_upstream_is_504(edge, monkeypatch):
    import src.server as server_mod
    monkeypatch.setattr(server_mod, "SPEAK_UPSTREAM_TIMEOUT", 0.2)
    edge.upstream.delay = 1.0
    r = _speak(edge, text="hello there")
    assert r.status_code == 504 and "timed out" in r.json()["detail"]
    assert _phrases(edge.paths) == []


def test_concurrent_misses_for_one_clip_fetch_it_once(edge):
    """A burst of pupils hitting the same new word must cost one generation."""
    edge.upstream.delay = 0.3
    with ThreadPoolExecutor(max_workers=6) as pool:
        responses = list(pool.map(lambda _: _speak(edge, text="hello there", voice="Puck"), range(6)))
    assert all(r.status_code == 200 for r in responses)
    assert len({r.content for r in responses}) == 1
    assert len(edge.upstream.requests) == 1
    assert sorted(r.headers["x-source"] for r in responses) == ["bank"] * 5 + ["upstream"]


def test_health_counts_upstream_as_a_way_to_serve(edge):
    body = edge.get("/health").json()
    assert body["healthy"] is True and body["upstream"] is True and body["bank"] is False


@pytest.mark.parametrize("token,warning", [
    (None, "SPEAK_UPSTREAM_TOKEN is not"),
    ("garbage", "not a readable JWT"),
    (jwt.encode({"exp": int(time.time()) + 86400}, "s" * 32, algorithm="HS256"), "expires in 1 days"),
    (jwt.encode({"exp": int(time.time()) - 86400}, "s" * 32, algorithm="HS256"), "expired"),
])
def test_startup_warns_about_a_missing_or_expiring_upstream_token(bank_env, monkeypatch, caplog,
                                                                  token, warning):
    monkeypatch.setenv("SPEAK_UPSTREAM_URL", "http://upstream.invalid")
    if token is None:
        monkeypatch.delenv("SPEAK_UPSTREAM_TOKEN", raising=False)
    else:
        monkeypatch.setenv("SPEAK_UPSTREAM_TOKEN", token)
    from src.server import app
    with caplog.at_level(logging.WARNING, logger="uvicorn.error"), TestClient(app):
        pass
    assert warning in caplog.text


_UPSTREAM_SERVER = """
import sys
import wordbank.gemini_tts as g
g.synthesize_pcm = lambda text, voice="Kore", instruct="", model=None: bytes(range(256)) * 200
import uvicorn
uvicorn.run("src.server:app", host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
"""


@pytest.mark.parametrize("store", ["1", "0"], ids=["upstream-stores", "upstream-stateless"])
def test_edge_against_a_real_upstream_server(bank_env, fake_gemini, tmp_path, monkeypatch, store):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    secret = "upstream-secret-0123456789abcdef0123456789"
    up_dir = tmp_path / "upstream"
    env = {**os.environ, "BANK_DB": str(up_dir / "wordbank.db"), "OGG_ROOT": str(up_dir / "ogg"),
           "WAV_ROOT": str(up_dir / "wav"), "JWT_SECRET": secret, "GEMINI_API_KEY": "fake",
           "SPEAK_STORE": store}
    env.pop("SPEAK_UPSTREAM_URL", None)
    proc = subprocess.Popen([sys.executable, "-c", _UPSTREAM_SERVER, str(port)], cwd=REPO_ROOT,
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(url + "/health", timeout=1)
                break
            except OSError:
                time.sleep(0.1)
        else:
            pytest.fail(f"upstream didn't start: {proc.stdout.read1().decode()}")

        monkeypatch.setenv("SPEAK_UPSTREAM_URL", url)
        monkeypatch.setenv("SPEAK_UPSTREAM_TOKEN", "not-a-valid-token")
        from src.server import app
        with TestClient(app) as c:
            r = _speak(c, text="hello there", voice="Puck")
            assert r.status_code == 502 and "rejected the edge's token" in r.json()["detail"]

            monkeypatch.setenv("SPEAK_UPSTREAM_TOKEN", jwt.encode(
                {"sub": "edge", "exp": int(time.time()) + 3600}, secret, algorithm="HS256"))
            miss = _speak(c, text="hello there", voice="Puck")
            assert miss.status_code == 200, miss.text
            assert miss.headers["x-source"] == "upstream" and miss.headers["x-voice"] == "gemini-Puck"
            assert miss.content[:4] == b"OggS"
            hit = _speak(c, text="hello there", voice="Puck")
            assert hit.headers["x-source"] == "bank" and hit.content == miss.content

        found = urllib.request.urlopen(url + "/v1/search?prefix=hello").read()
        if store == "1":
            # upstream banked it too, so a second edge would get it without Gemini
            assert b"gemini-Puck" in found
        else:
            # stateless upstream: the clip lives only on the edge
            assert b"gemini-Puck" not in found
            assert not up_dir.exists()
        assert fake_gemini.calls == []                  # the edge process never called Gemini
    finally:
        proc.terminate()
        proc.wait(timeout=10)
