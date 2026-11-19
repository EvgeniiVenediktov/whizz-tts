"""Phrase bank: separate storage from words, voice on results, merged search,
type flag, and phrase serving."""
from fastapi.testclient import TestClient
import pytest

from tests.conftest import AUDIO
from wordbank.lookup import BankLookup
from wordbank.manifest import DONE, Manifest


def _phrase_manifest(db):
    return Manifest(db, table="phrases", text_col="text")


# ---- storage: separate tables, no overlap ----

def test_words_and_phrases_are_separate(tmp_path):
    db = str(tmp_path / "wb.db")
    wm, pm = Manifest(db), _phrase_manifest(db)
    wm.add_words(["cat"], "en-us", voice="Kore")
    pm.add_words(["the cat sat"], "en-us", voice="en-gb-tutor-11")

    # Independent tables: each holds only its own kind.
    assert [r["word"] for r in wm.iter_rows()] == ["cat"]
    prow = pm.iter_rows()[0]
    assert prow["text"] == "the cat sat" and prow["voice"] == "en-gb-tutor-11"
    wm.close()
    pm.close()


def test_phrase_unique_per_voice(tmp_path):
    pm = _phrase_manifest(str(tmp_path / "wb.db"))
    assert pm.add_words(["hello world"], "en-us", voice="en-gb-tutor-11") == 1
    assert pm.add_words(["hello world"], "en-us", voice="Puck") == 1  # diff voice → new
    assert pm.add_words(["hello world"], "en-us", voice="en-gb-tutor-11") == 0  # dup → ignored
    assert sum(pm.counts().values()) == 2
    pm.close()


# ---- lookup: merged search, voice, type, phrase get ----

def _seed(db):
    wm, pm = Manifest(db), _phrase_manifest(db)
    wm.add_words(["cat", "camel"], "en-us", voice="Kore")
    ids = {r["word"]: r["id"] for r in wm.iter_rows()}
    wm.set_status(ids["cat"], DONE, file_path="en-us/gemini-Kore/0/1_cat.wav", voice="Kore")
    # camel left pending → not ready
    pm.add_words(["cathedral bells"], "en-us", voice="en-gb-tutor-11")
    pid = pm.iter_rows()[0]["id"]
    pm.set_status(pid, DONE, file_path="en-us/gemini-en-gb-tutor-11/phrases/0/1_cathedral.wav", voice="en-gb-tutor-11")
    wm.close()
    pm.close()


def test_search_merges_and_tags(tmp_path):
    db = str(tmp_path / "wb.db")
    _seed(db)
    bl = BankLookup(db)

    res = {r["text"]: r for r in bl.search("ca", "en-us", 10)}
    assert res["cat"]["type"] == "word" and res["cat"]["voice"] == "Kore"
    assert res["cathedral bells"]["type"] == "phrase" and res["cathedral bells"]["voice"] == "en-gb-tutor-11"
    assert "camel" not in res                              # pending → excluded

    assert all(r["type"] == "word" for r in bl.search("ca", "en-us", 10, kind="word"))
    assert all(r["type"] == "phrase" for r in bl.search("ca", "en-us", 10, kind="phrase"))

    assert bl.get("cat", "en-us")["voice"] == "Kore"       # voice on word result
    ph = bl.get_phrase("cathedral bells", "en-us")
    assert ph["voice"] == "en-gb-tutor-11" and ph["file_path"].endswith("1_cathedral.wav")
    assert bl.get_phrase("nope", "en-us") is None
    assert bl.phrase_languages() == [{"code": "en-us", "words": 1}]
    bl.close()


@pytest.fixture
def search_client(tmp_path, monkeypatch):
    """TestClient wired to a bank with one word (`cat`) and one phrase
    (`hello world`), both with WAV+OGG on disk — shared setup for the
    /v1/search, /v1/phrase*, /v1/bank/*/info, and /v1/languages tests below."""
    wav_root, ogg_root = tmp_path / "words", tmp_path / "words_ogg"
    for rel in ("en-us/gemini-Kore/0/1_cat.wav", "en-us/gemini-en-gb-tutor-11/phrases/0/1_hello.wav"):
        (wav_root / rel).parent.mkdir(parents=True, exist_ok=True)
        (wav_root / rel).write_bytes(b"RIFF" + b"\x00" * 100)
        ogg = ogg_root / (rel[:-4] + ".ogg")
        ogg.parent.mkdir(parents=True, exist_ok=True)
        ogg.write_bytes(b"OggS" + b"\x00" * 100)

    db = str(tmp_path / "wb.db")
    wm, pm = Manifest(db), _phrase_manifest(db)
    wm.add_words(["cat"], "en-us", voice="Kore")
    wm.set_status(wm.iter_rows()[0]["id"], DONE,
                  file_path="en-us/gemini-Kore/0/1_cat.wav", voice="Kore")
    pm.add_words(["hello world"], "en-us", voice="en-gb-tutor-11")
    pm.set_status(pm.iter_rows()[0]["id"], DONE,
                  file_path="en-us/gemini-en-gb-tutor-11/phrases/0/1_hello.wav", voice="en-gb-tutor-11")
    wm.close()
    pm.close()

    monkeypatch.setenv("BANK_DB", db)
    monkeypatch.setenv("WAV_ROOT", str(wav_root))
    monkeypatch.setenv("OGG_ROOT", str(ogg_root))
    from src.server import app
    with TestClient(app) as c:
        yield c


def test_search_returns_words_and_phrases(search_client):
    words = search_client.get("/v1/search", params={"prefix": "ca", "lang": "en-us"}).json()["results"]
    assert {"text": "cat", "type": "word", "voice": "Kore"} in words

    phrases = search_client.get("/v1/search", params={"prefix": "hel", "lang": "en-us"}).json()["results"]
    assert any(p["text"] == "hello world" and p["type"] == "phrase"
               and p["voice"] == "en-gb-tutor-11" for p in phrases)


def test_phrase_info_endpoint(search_client):
    pinfo = search_client.get("/v1/phrase/info", params={"text": "hello world", "lang": "en-us"}).json()
    assert pinfo["type"] == "phrase" and pinfo["voice"] == "en-gb-tutor-11" and pinfo["ogg"]["exists"]


def test_phrase_audio_endpoint(search_client):
    assert search_client.get("/v1/phrase", params={"text": "hello world", "lang": "en-us"}).content[:4] == b"OggS"
    assert search_client.get("/v1/phrase", params={"text": "nope", "lang": "en-us"}).status_code == 404


def test_word_info_endpoint_includes_type_and_voice(search_client):
    winfo = search_client.get("/v1/bank/cat/info").json()
    assert winfo["type"] == "word" and winfo["voice"] == "Kore"


def test_languages_includes_phrase_langs(search_client):
    assert {"code": "en-us", "words": 1} in search_client.get("/v1/languages").json()["phrasebank"]


def test_search_prefix_is_literal_and_limit_is_clamped(client):
    """`%`/`_` in a prefix match themselves, limit stays in 1..50 (SQLite reads
    LIMIT -1 as no limit), and a prefix ending in U+10FFFF (no range bound
    above it) is a miss, not a 500."""
    for voice in ("Kore", "Puck"):
        client.post("/v1/speak", headers=AUDIO, data={"text": "hello there", "voice": voice})

    def search(prefix, **params):
        r = client.get("/v1/search", params={"prefix": prefix, **params})
        assert r.status_code == 200
        return r.json()["results"]
    assert len(search("hello")) == 2 and len(search("hello", limit=-1)) == 1
    assert search("%") == search("_ello") == search("hello\U0010ffff") == []
    assert client.get("/v1/bank", params={"prefix": "\U0010ffff"}).json() == {"words": []}
