"""`lang` from a request is the first directory of every clip the server saves
(`<lang>/<voice>/…`), so it is validated before anything else happens: a bad
code is a 400 with no lookup, synthesis, upstream fetch, file, or row. The
writers also refuse any path that resolves outside their root (defence in
depth, tested directly). All roots are under tmp_path (bank_env); Gemini, the
model and upstream are faked."""
import os
import sqlite3
from contextlib import contextmanager

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from tests.conftest import make_wav_bytes
from tests.test_edge import FakeUpstream
from wordbank.languages import is_lang_code, normalize_lang
from wordbank.manifest import Manifest
from wordbank.voices import path_under, save_voice_clip, voice_rel_path

# "{out}" is filled with a directory outside the tmp bank but inside pytest's
# basetemp (which _files watches), so a regression can't write to the real /tmp.
BAD_LANGS = [
    "../x", "../../escaped", "en/../..", "en-us/../../x", "{out}/abs-escape",
    "..", ".", "  ", "en_us", "e", "en-us\x00",
    "ру-ру",            # non-ASCII
    "KK",     # Kelvin signs: lowercase to ASCII "kk", refused before folding
]
# What a path segment can carry (the voice-audio route's {lang}).
SEGMENT_BAD_LANGS = [l for l in BAD_LANGS if "/" not in l]
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


def _save_voice(client, name, lang, wav):
    return client.post("/v1/voices", data={"name": name, "lang": lang, "ref_text": "hi there"},
                       files={"audio": (f"{name}.wav", wav, "audio/wav")})


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

@pytest.mark.parametrize("lang", BAD_LANGS)
def test_speak_rejects_bad_lang_before_generating(client, tmp_path, lang):
    with _nothing_written(tmp_path, client.paths.db):
        r = client.post("/v1/speak", data={"text": "hello there", "lang": _bad(lang, tmp_path)})
    assert r.status_code == 400 and "invalid lang" in r.json()["detail"]
    assert client.gemini.calls == []


@pytest.mark.parametrize("lang", ["../../escaped", "{out}/abs-escape", "ру-ру"])
@pytest.mark.parametrize("mode", ["instruct", "stateless"])
def test_speak_one_off_paths_reject_bad_lang_too(client, tmp_path, monkeypatch, lang, mode):
    """instruct and SPEAK_STORE=0 skip the bank but still call Gemini."""
    data = {"text": "hello there", "lang": _bad(lang, tmp_path)}
    if mode == "instruct":
        data["instruct"] = "whisper"
    else:
        monkeypatch.setenv("SPEAK_STORE", "0")
    with _nothing_written(tmp_path, client.paths.db):
        r = client.post("/v1/speak", data=data)
    assert r.status_code == 400 and client.gemini.calls == []


@pytest.mark.parametrize("lang", BAD_LANGS)
def test_edge_speak_rejects_bad_lang_before_fetching_upstream(edge, tmp_path, lang):
    with _nothing_written(tmp_path, edge.paths.db):
        r = edge.post("/v1/speak", data={"text": "hello there", "lang": _bad(lang, tmp_path)})
    assert r.status_code == 400 and "invalid lang" in r.json()["detail"]
    assert edge.upstream.requests == [] and edge.gemini.calls == []


@pytest.mark.parametrize("label", ["../../escaped", "a/b", "..", "x\\y"])
def test_edge_refuses_an_upstream_voice_label_that_is_not_one_segment(edge, tmp_path, label):
    """The upstream's X-Voice becomes a directory of the saved clip."""
    edge.upstream.respond = lambda f: (200, {"X-Source": "gemini", "X-Voice": label}, b"OggS-x")
    with _nothing_written(tmp_path, edge.paths.db):
        r = edge.post("/v1/speak", data={"text": "hello there"})
    assert r.status_code == 502 and "unusable voice label" in r.json()["detail"]


@pytest.mark.parametrize("lang", BAD_LANGS)
def test_save_voice_rejects_bad_lang(client, tmp_path, lang):
    with _nothing_written(tmp_path, client.paths.db):
        r = _save_voice(client, "kore", _bad(lang, tmp_path), make_wav_bytes(0.2))
    assert r.status_code == 400 and "invalid lang" in r.json()["detail"]


@pytest.mark.parametrize("lang", BAD_LANGS)
@pytest.mark.parametrize("voice", [None, "kore"])
def test_tts_rejects_bad_lang_before_synthesis(client, tmp_path, lang, voice):
    assert _save_voice(client, "kore", "en-us", make_wav_bytes(0.2)).status_code == 200
    data = {"text": "hello", "lang": _bad(lang, tmp_path), **({"voice": voice} if voice else {})}
    with _nothing_written(tmp_path, client.paths.db):
        r = client.post("/v1/tts", data=data)
    assert r.status_code == 400 and "invalid lang" in r.json()["detail"]
    assert client.model.calls == []


@pytest.mark.parametrize("lang", SEGMENT_BAD_LANGS)
def test_voice_audio_rejects_bad_lang(client, lang):
    assert _save_voice(client, "kore", "en-us", make_wav_bytes(0.2)).status_code == 200
    seg = "".join(f"%{b:02X}" for b in lang.encode())   # every byte, so '..' isn't collapsed client-side
    r = client.get(f"/v1/voices/{seg}/kore/audio")
    assert r.status_code == 400 and "invalid lang" in r.json()["detail"]


@pytest.mark.parametrize("lang", ["", "../x", "ру-ру"])
@pytest.mark.parametrize("path,params", [
    ("/v1/search", {"prefix": "he"}), ("/v1/bank", {"prefix": "he"}),
    ("/v1/bank/hello", {}), ("/v1/bank/hello/info", {}),
    ("/v1/phrase", {"text": "hello there"}), ("/v1/phrase/info", {"text": "hello there"}),
    ("/v1/voices", {}),
])
def test_read_routes_reject_bad_lang(client, path, params, lang):
    r = client.get(path, params={**params, "lang": lang})
    assert r.status_code == 400 and "invalid lang" in r.json()["detail"]


# ---- valid langs still work end to end ----

@pytest.mark.parametrize("lang", VALID_LANGS)
def test_valid_lang_speak_saves_under_it_and_then_serves_from_bank(client, lang):
    r = client.post("/v1/speak", data={"text": "hello there", "lang": lang})
    assert r.status_code == 200 and r.headers["x-source"] == "gemini"
    m = Manifest(str(client.paths.db), table="phrases", text_col="text")
    [row] = [dict(x) for x in m.iter_rows()]
    m.close()
    assert row["lang"] == lang and row["file_path"].startswith(f"{lang}/gemini-Kore/phrases/")
    assert (client.paths.ogg / (row["file_path"][:-4] + ".ogg")).read_bytes() == r.content

    again = client.post("/v1/speak", data={"text": "hello there", "lang": lang})
    assert again.headers["x-source"] == "bank" and again.content == r.content
    assert len(client.gemini.calls) == 1


@pytest.mark.parametrize("lang", VALID_LANGS)
def test_valid_lang_voice_saves_serves_and_clones(client, lang):
    wav = make_wav_bytes(0.2)
    r = _save_voice(client, "narrator", lang, wav)
    assert r.status_code == 200 and r.json()["ref_audio"] == f"{lang}/narrator.wav"
    assert (client.paths.voices / lang / "narrator.wav").read_bytes() == wav
    assert client.get(f"/v1/voices/{lang}/narrator/audio").content == wav
    assert [v["lang"] for v in client.get("/v1/voices", params={"lang": lang}).json()["voices"]] == [lang]

    r = client.post("/v1/tts", data={"text": "hello", "voice": "narrator", "lang": lang})
    assert r.status_code == 200
    assert client.model.calls[0]["ref_bytes"] == wav


def test_lang_is_case_and_space_insensitive_and_filed_lowercase(client):
    """Normalized, not rejected: "EN-US" finds the existing "en-us" rows."""
    first = client.post("/v1/speak", data={"text": "hello there", "lang": "EN-US"})
    assert first.status_code == 200
    for lang in ("en-us", " En-Us "):
        r = client.post("/v1/speak", data={"text": "hello there", "lang": lang})
        assert r.headers["x-source"] == "bank" and r.content == first.content
    assert client.get("/v1/phrase", params={"text": "hello there", "lang": "EN-us"}).content == first.content
    m = Manifest(str(client.paths.db), table="phrases", text_col="text")
    assert [x["lang"] for x in m.iter_rows()] == ["en-us"]
    m.close()

    wav = make_wav_bytes(0.2)
    r = _save_voice(client, "narrator", "RU-RU", wav)
    assert r.json()["lang"] == "ru-ru" and r.json()["ref_audio"] == "ru-ru/narrator.wav"
    assert client.get("/v1/voices/RU-RU/narrator/audio").content == wav


def test_empty_form_lang_means_the_default(client):
    """FastAPI hands an empty form field the parameter's default (DEFAULT_LANG)."""
    assert client.post("/v1/speak", data={"text": "hello there", "lang": ""}).status_code == 200
    m = Manifest(str(client.paths.db), table="phrases", text_col="text")
    assert [x["lang"] for x in m.iter_rows()] == ["en-us"]
    m.close()


def test_edge_forwards_and_files_the_normalized_lang(edge):
    r = edge.post("/v1/speak", data={"text": "hello there", "lang": "RU-RU"})
    assert r.status_code == 200 and r.headers["x-source"] == "upstream"
    assert edge.upstream.requests[0]["lang"] == "ru-ru"
    m = Manifest(str(edge.paths.db), table="phrases", text_col="text")
    [row] = [dict(x) for x in m.iter_rows()]
    m.close()
    assert row["lang"] == "ru-ru" and row["file_path"].startswith("ru-ru/gemini-Kore/phrases/")


# ---- the validator itself ----

def test_normalize_lang_accepts_codes_and_returns_the_stored_form():
    assert [normalize_lang(c) for c in ("en-us", "EN-GB", " ru-RU\n", "fr", "zh-hant-tw", "ar-xa")] \
        == ["en-us", "en-gb", "ru-ru", "fr", "zh-hant-tw", "ar-xa"]


@pytest.mark.parametrize("lang", BAD_LANGS + ["", None, "/tmp/x", "en-" + "-".join(["abcdefgh"] * 4)])
def test_normalize_lang_rejects_non_codes(lang):
    with pytest.raises(ValueError, match="invalid lang"):
        normalize_lang(lang)


def test_is_lang_code_wants_the_canonical_form():
    assert is_lang_code("en-us") and is_lang_code("ru-ru")
    assert not any(is_lang_code(c) for c in ("EN-US", " en-us", "en-us\n", "", None))


# ---- defence in depth: wordbank.voices and the /v1/speak writer ----

@pytest.mark.parametrize("lang", BAD_LANGS + ["EN-US", "/tmp/x"])
def test_voice_rel_path_refuses_bad_lang(lang):
    """No sanitizing here: a non-canonical code is refused (callers normalize)."""
    with pytest.raises(ValueError, match="invalid lang"):
        voice_rel_path(lang, "kore")


@pytest.mark.parametrize("lang", ["../../escaped", "{out}/abs-escape", "en/../.."])
def test_save_voice_clip_refuses_bad_lang_and_writes_nothing(tmp_path, lang):
    """The CLI (`add-voice` / `gemini-voice`) reaches this without the server."""
    db = tmp_path / "wb.db"
    m = Manifest(str(db))
    try:
        with _nothing_written(tmp_path, db):
            with pytest.raises(ValueError):
                save_voice_clip(m, str(tmp_path / "bank" / "voices"), "kore",
                                _bad(lang, tmp_path), b"RIFF", "hi")
        assert m.voices() == []
    finally:
        m.close()


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


def test_save_voice_clip_refuses_a_lang_dir_symlinked_out_of_the_root(tmp_path):
    """A valid lang can still resolve elsewhere; the containment check catches it."""
    root, outside = tmp_path / "voices", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "en-us").symlink_to(outside)
    m = Manifest(str(tmp_path / "wb.db"))
    try:
        with pytest.raises(ValueError, match="outside"):
            save_voice_clip(m, str(root), "kore", "en-us", b"RIFF", "hi")
        assert list(outside.iterdir()) == [] and m.voices() == []
    finally:
        m.close()


@pytest.mark.parametrize("lang,voice", [("../../escaped", "gemini-Kore"),
                                        ("{out}", "gemini-Kore"),
                                        ("en-us", "../../../escaped")])
def test_save_phrase_refuses_a_path_outside_the_bank(client, tmp_path, lang, voice):
    """Called directly, past the request validation: the writer's own check."""
    from src.server import _save_phrase
    with _nothing_written(tmp_path, client.paths.db):
        with pytest.raises(HTTPException) as exc:
            _save_phrase("hello there", _bad(lang, tmp_path), voice, b"OggS", pcm=b"\0\0")
    assert exc.value.status_code == 500 and "outside" in exc.value.detail
