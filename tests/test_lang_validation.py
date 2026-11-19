"""`lang` from a request is the first directory of every clip the server saves
(`<lang>/<voice>/…`), so it is validated before anything else happens: a bad
code is a 400 with no lookup, synthesis, upstream fetch, file, or row. The
writers also refuse any path that resolves outside their root (defence in
depth, tested directly). All roots are under tmp_path (bank_env); Gemini and
upstream are faked."""
import os
import sqlite3
from contextlib import contextmanager

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from tests.conftest import AUDIO
from tests.test_edge import FakeUpstream
from wordbank.languages import is_lang_code, normalize_lang
from wordbank.manifest import Manifest
from wordbank.paths import path_under

# "{out}" is filled with a directory outside the tmp bank but inside pytest's
# basetemp (which _files watches), so a regression can't write to the real /tmp.
BAD_LANGS = [
    "../x", "../../escaped", "en/../..", "en-us/../../x", "{out}/abs-escape",
    "..", ".", "  ", "en_us", "e", "en-us\x00",
    "ру-ру",            # non-ASCII
    "KK",     # Kelvin signs: lowercase to ASCII "kk", refused before folding
]
# Every shape above is checked once against the validator itself; each route is
# then checked with the two that would escape the bank (a relative traversal and
# an absolute path), which proves the route runs the validator.
ESCAPES = ["../../escaped", "{out}/abs-escape"]
VALID_LANGS = ["en-us", "en-gb", "ru-ru"]


def _bad(lang, tmp_path):
    return lang.format(out=tmp_path.parent / f"out-{tmp_path.name}")


def _files(tmp_path, db):
    """Every file a '../' or absolute lang could reach from the tmp bank, i.e.
    all of pytest's basetemp. The bank DB is left out (SQLite opens/drops its
    -wal/-shm on its own); _rows covers it."""
    return {os.path.join(d, f) for d, _, fs in os.walk(tmp_path.parent) for f in fs
            if not os.path.join(d, f).startswith(str(db))}


def _rows(db):
    """Row count per table; {} while the bank doesn't exist."""
    if not os.path.exists(db):
        return {}
    conn = sqlite3.connect(db)
    try:
        tables = [t for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
    finally:
        conn.close()


@contextmanager
def _nothing_written(tmp_path, db):
    rows, files = _rows(db), _files(tmp_path, db)
    yield
    assert _files(tmp_path, db) == files, "a file was written"
    assert _rows(db) == rows, "a bank row was written"


@pytest.fixture(scope="module")
def _upstream():
    up = FakeUpstream()      # one per module: shutdown() alone costs 0.5 s
    yield up
    up.server.shutdown()


@pytest.fixture
def edge(bank_env, fake_gemini, _upstream, monkeypatch):
    _upstream.requests, _upstream.respond = [], _upstream._ok
    monkeypatch.setenv("SPEAK_UPSTREAM_URL", _upstream.url)
    from src.server import app
    with TestClient(app) as c:
        c.upstream, c.gemini, c.paths = _upstream, fake_gemini, bank_env
        yield c


# ---- bad lang: 400, before anything is looked up, generated, or written ----

def test_speak_rejects_bad_lang_before_generating(client, tmp_path):
    for lang in ESCAPES:
        with _nothing_written(tmp_path, client.paths.db):
            r = client.post("/v1/speak", headers=AUDIO, data={"text": "hello there", "lang": _bad(lang, tmp_path)})
        assert r.status_code == 400 and "invalid lang" in r.json()["detail"], lang
    assert client.gemini.calls == []


def test_speak_one_off_paths_reject_bad_lang_too(client, tmp_path, monkeypatch):
    """instruct and SPEAK_STORE=0 skip the bank but still call Gemini."""
    bad = {"text": "hello there", "lang": _bad(ESCAPES[0], tmp_path)}
    with _nothing_written(tmp_path, client.paths.db):
        assert client.post("/v1/speak", headers=AUDIO, data={**bad, "instruct": "whisper"}).status_code == 400
        monkeypatch.setenv("SPEAK_STORE", "0")
        assert client.post("/v1/speak", headers=AUDIO, data=bad).status_code == 400
    assert client.gemini.calls == []


def test_edge_speak_rejects_bad_lang_before_fetching_upstream(edge, tmp_path):
    for lang in ESCAPES:
        with _nothing_written(tmp_path, edge.paths.db):
            r = edge.post("/v1/speak", headers=AUDIO, data={"text": "hello there", "lang": _bad(lang, tmp_path)})
        assert r.status_code == 400 and "invalid lang" in r.json()["detail"], lang
    assert edge.upstream.requests == [] and edge.gemini.calls == []


def test_edge_refuses_an_upstream_voice_that_is_not_a_known_voice(edge, tmp_path):
    """The upstream's X-Voice becomes a directory of the saved clip and is
    shown by the frontend: only a known voice is taken."""
    for label in ("../../escaped", "a/b", "..", "<img src=x onerror=alert(1)>", "k" * 300):
        edge.upstream.respond = lambda f, label=label: (200, {"X-Source": "gemini", "X-Voice": label}, b"OggS-x")
        with _nothing_written(tmp_path, edge.paths.db):
            r = edge.post("/v1/speak", headers=AUDIO, data={"text": "hello there"})
        assert r.status_code == 502 and r.json()["detail"] == "upstream reported an unknown voice", label


def test_edge_refuses_an_upstream_clip_id_that_is_not_a_uuid(edge, tmp_path):
    """It names the saved file of a long text."""
    edge.upstream.respond = lambda f: (200, {"X-Voice": "Kore", "X-Clip-Id": "../../escaped"}, b"OggS")
    with _nothing_written(tmp_path, edge.paths.db):
        r = edge.post("/v1/speak", data={"text": "a text long enough to be filed by its clip id"})
    assert r.status_code == 502 and r.json()["detail"] == "upstream reported an invalid clip id"


def test_read_routes_reject_bad_lang(client):
    for path, params in [
        ("/v1/search", {"prefix": "he"}), ("/v1/bank", {"prefix": "he"}),
        ("/v1/bank/hello", {}), ("/v1/bank/hello/info", {}),
        ("/v1/phrase", {"text": "hello there"}), ("/v1/phrase/info", {"text": "hello there"}),
    ]:
        r = client.get(path, params={**params, "lang": "../x"})
        assert r.status_code == 400 and "invalid lang" in r.json()["detail"], path


# ---- valid langs still work end to end ----

def test_valid_lang_speak_saves_under_it_and_then_serves_from_bank(client):
    lang = "en-gb"
    r = client.post("/v1/speak", headers=AUDIO, data={"text": "hello there", "lang": lang})
    assert r.status_code == 200 and r.headers["x-source"] == "gemini"
    m = Manifest(str(client.paths.db), table="phrases", text_col="text")
    [row] = [dict(x) for x in m.iter_rows()]
    m.close()
    assert row["lang"] == lang and row["file_path"].startswith(f"{lang}/gemini-Kore/phrases/")
    assert (client.paths.ogg / (row["file_path"][:-4] + ".ogg")).read_bytes() == r.content

    again = client.post("/v1/speak", headers=AUDIO, data={"text": "hello there", "lang": lang})
    assert again.headers["x-source"] == "bank" and again.content == r.content
    assert len(client.gemini.calls) == 1


def test_lang_is_case_and_space_insensitive_and_filed_lowercase(client):
    """Normalized, not rejected: "EN-US" finds the existing "en-us" rows."""
    first = client.post("/v1/speak", headers=AUDIO, data={"text": "hello there", "lang": "EN-US"})
    assert first.status_code == 200
    for lang in ("en-us", " En-Us "):
        r = client.post("/v1/speak", headers=AUDIO, data={"text": "hello there", "lang": lang})
        assert r.headers["x-source"] == "bank" and r.content == first.content
    assert client.get("/v1/phrase", params={"text": "hello there", "lang": "EN-us"}).content == first.content
    m = Manifest(str(client.paths.db), table="phrases", text_col="text")
    assert [x["lang"] for x in m.iter_rows()] == ["en-us"]
    m.close()


def test_empty_form_lang_means_the_default(client):
    """FastAPI hands an empty form field the parameter's default (DEFAULT_LANG)."""
    assert client.post("/v1/speak", headers=AUDIO, data={"text": "hello there", "lang": ""}).status_code == 200
    m = Manifest(str(client.paths.db), table="phrases", text_col="text")
    assert [x["lang"] for x in m.iter_rows()] == ["en-us"]
    m.close()


def test_edge_forwards_and_files_the_normalized_lang(edge):
    r = edge.post("/v1/speak", headers=AUDIO, data={"text": "hello there", "lang": "RU-RU"})
    assert r.status_code == 200 and r.headers["x-source"] == "upstream"
    assert edge.upstream.requests[0]["lang"] == "ru-ru"
    m = Manifest(str(edge.paths.db), table="phrases", text_col="text")
    [row] = [dict(x) for x in m.iter_rows()]
    m.close()
    assert row["lang"] == "ru-ru" and row["file_path"].startswith("ru-ru/gemini-Kore/phrases/")


# ---- the validator itself ----

def test_normalize_lang_accepts_codes_and_returns_the_stored_form():
    assert [normalize_lang(c) for c in VALID_LANGS + ["EN-GB", " ru-RU\n", "fr", "zh-hant-tw", "ar-xa"]] \
        == VALID_LANGS + ["en-gb", "ru-ru", "fr", "zh-hant-tw", "ar-xa"]


def test_normalize_lang_rejects_non_codes():
    for lang in BAD_LANGS + ["", None, "/tmp/x", "en-" + "-".join(["abcdefgh"] * 4)]:
        with pytest.raises(ValueError, match="invalid lang"):
            normalize_lang(lang)


def test_is_lang_code_wants_the_canonical_form():
    assert is_lang_code("en-us") and is_lang_code("ru-ru")
    assert not any(is_lang_code(c) for c in ("EN-US", " en-us", "en-us\n", "", None))


# ---- defence in depth: the /v1/speak writer ----

def test_path_under(tmp_path):
    root = tmp_path / "root"
    (tmp_path / "outside").mkdir()
    root.mkdir()
    (root / "out").symlink_to(tmp_path / "outside")
    assert path_under(str(root), "en-us/x.wav") == os.path.join(str(root), "en-us/x.wav")
    for rel in ("../x", "en-us/../../x", str(tmp_path / "outside" / "x"), "out/x.wav"):
        with pytest.raises(ValueError, match="outside"):
            path_under(str(root), rel)
    (tmp_path / "linked-root").symlink_to(root)          # a symlinked root itself is fine
    assert path_under(str(tmp_path / "linked-root"), "en-us/x.wav")


def test_save_phrase_refuses_a_path_outside_the_bank(client, tmp_path):
    """Called directly, past the request validation: the writer's own check
    (a bad lang, an absolute lang, and a bad voice label)."""
    from src.server import _Clip, _save_phrase
    for lang, voice in [("../../escaped", "Kore"), ("{out}", "Kore"), ("en-us", "x/../../../escaped")]:
        with _nothing_written(tmp_path, client.paths.db):
            with pytest.raises(HTTPException) as exc:
                _save_phrase("hello there", _bad(lang, tmp_path), _Clip(voice, "gemini", ogg=b"OggS", pcm=b"\0\0"))
        assert exc.value.status_code == 500 and "outside" in exc.value.detail, (lang, voice)
