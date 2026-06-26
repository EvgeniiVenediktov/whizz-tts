"""Tests for the lightweight bank lookup + server bank endpoints."""
from fastapi.testclient import TestClient

from wordbank.gemini_batch import write_wav
from wordbank.lookup import BankLookup
from wordbank.manifest import DONE, Manifest
from wordbank.paths import mirror_path


def _bank_db(tmp_path, entries):
    """entries: list of (word, status, file_path)."""
    m = Manifest(str(tmp_path / "wb.db"))
    m.add_words([w for w, _, _ in entries], "en")
    ids = {r["word"]: r["id"] for r in m.iter_rows()}
    for word, status, fp in entries:
        m.set_status(ids[word], status, file_path=fp)
    m.close()
    return str(tmp_path / "wb.db")


# --- lookup unit ---

def test_lookup_exact_case_insensitive(tmp_path):
    db = _bank_db(tmp_path, [("cat", DONE, "/x/cat.wav"), ("dog", DONE, "/x/dog.wav")])
    bank = BankLookup(db)
    assert bank.get("cat")["file_path"] == "/x/cat.wav"
    assert bank.get("CAT")["file_path"] == "/x/cat.wav"
    assert bank.get("missing") is None
    bank.close()


def test_lookup_ignores_unfinished(tmp_path):
    db = _bank_db(tmp_path, [("pend", "pending", None), ("fail", "failed", None)])
    bank = BankLookup(db)
    assert bank.get("pend") is None and bank.get("fail") is None
    bank.close()


def test_suggest_prefix_only_and_sorted(tmp_path):
    db = _bank_db(tmp_path, [(w, DONE, f"/{w}.wav") for w in ("car", "care", "card", "dog")])
    bank = BankLookup(db)
    assert bank.suggest("car") == ["car", "card", "care"]
    assert bank.suggest("xyz") == []
    bank.close()


def test_lookup_missing_db_is_tolerant(tmp_path):
    bank = BankLookup(str(tmp_path / "nope.db"))
    assert not bank.available
    assert bank.get("cat") is None and bank.suggest("c") == []
    assert bank.languages() == []


def test_search_is_language_scoped(tmp_path):
    """Lookups must isolate by language — the core multi-language feature."""
    db = str(tmp_path / "wb.db")
    m = Manifest(db)
    m.add_words(["cat", "car"], "en-us")
    m.add_words(["camion", "car"], "fr")        # "car" in both; "camion" fr-only
    ids = {(r["word"], r["lang"]): r["id"] for r in m.iter_rows()}
    for key in ids:
        m.set_status(ids[key], DONE, file_path="x")
    m.close()

    bank = BankLookup(db)
    en = bank.suggest("ca", lang="en-us")
    fr = bank.suggest("ca", lang="fr")
    assert "cat" in en and "camion" not in en   # en-us search → only en-us words
    assert "camion" in fr and "cat" not in fr    # fr search → only fr words
    assert bank.get("camion", "en-us") is None   # exact get is lang-scoped too
    assert bank.get("camion", "fr") is not None


def test_languages(tmp_path):
    db = str(tmp_path / "wb.db")
    m = Manifest(db)
    m.add_words(["cat", "dog"], "en")
    m.add_words(["chat"], "fr")
    m.add_words(["pending"], "de")          # not done → its lang excluded
    ids = {(r["word"], r["lang"]): r["id"] for r in m.iter_rows()}
    m.set_status(ids[("cat", "en")], DONE, file_path="x")
    m.set_status(ids[("dog", "en")], DONE, file_path="x")
    m.set_status(ids[("chat", "fr")], DONE, file_path="x")
    m.close()

    langs = BankLookup(db).languages()
    assert {"code": "en", "words": 2} in langs
    assert {"code": "fr", "words": 1} in langs
    assert all(l["code"] != "de" for l in langs)   # pending lang not listed


def test_lookup_sees_rows_committed_after_open(tmp_path):
    """Regression: a long-lived reader must see words a writer commits later
    (the WAL/mode=ro private-shm bug that hid freshly-generated words)."""
    db = str(tmp_path / "wb.db")
    writer = Manifest(db)
    writer.add_words(["alpha"], "en")
    ids = {r["word"]: r["id"] for r in writer.iter_rows()}
    writer.set_status(ids["alpha"], DONE, file_path="/a.wav")

    bank = BankLookup(db)                     # reader opens now
    assert bank.get("alpha") is not None

    writer.add_words(["beta"], "en")          # writer commits AFTER reader opened
    ids = {r["word"]: r["id"] for r in writer.iter_rows()}
    writer.set_status(ids["beta"], DONE, file_path="/b.wav")
    assert bank.get("beta") is not None       # reader must see the new commit

    writer.close()
    bank.close()


# --- server endpoints ---

def test_bank_endpoints(tmp_path, monkeypatch):
    # Production ships OGG only — no WAV root mounted.
    ogg_root = tmp_path / "words_ogg"
    ogg = ogg_root / "en-us/gemini-Kore/0/1_cat.ogg"
    ogg.parent.mkdir(parents=True)
    ogg.write_bytes(b"OggS" + b"\x00" * 200)

    m = Manifest(str(tmp_path / "wb.db"))
    m.add_words(["cat", "camel"], "en-us")
    ids = {r["word"]: r["id"] for r in m.iter_rows()}
    m.set_status(ids["cat"], DONE, file_path="en-us/gemini-Kore/0/1_cat.wav")
    m.set_status(ids["camel"], DONE, file_path="en-us/gemini-Kore/0/2_camel.wav")  # no ogg
    m.close()

    monkeypatch.setenv("BANK_DB", str(tmp_path / "wb.db"))
    monkeypatch.setenv("OGG_ROOT", str(ogg_root))
    monkeypatch.setenv("WAV_ROOT", str(tmp_path / "no_wav"))   # WAV not shipped
    from src.server import app
    with TestClient(app) as c:
        r = c.get("/v1/bank/cat")                              # default = OGG
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("audio/ogg")
        assert r.content[:4] == b"OggS"

        # WAV requested but not shipped → 404 (production has no WAV)
        assert c.get("/v1/bank/cat", params={"format": "wav"}).status_code == 404

        assert c.get("/v1/bank/nope").status_code == 404       # not in bank
        assert c.get("/v1/bank/camel").status_code == 404      # row exists, no ogg

        words = c.get("/v1/bank", params={"prefix": "ca"}).json()["words"]
        assert "cat" in words and "camel" in words

        langs = c.get("/v1/languages").json()
        assert {"code": "en-us", "words": 2} in langs["wordbank"]      # data-driven (cat, camel)
        assert any(s["code"] == "en-us" for s in langs["generate"])    # curated list


# --- WAV/OGG mirror + compare ---

def test_mirror_path():
    assert mirror_path("output/words/en/gemini-Kore/0/1_the.wav",
                       "output/words", "output/words_ogg", ".ogg") \
        == "output/words_ogg/en/gemini-Kore/0/1_the.ogg"
    assert mirror_path("/elsewhere/x.wav", "output/words", "output/words_ogg", ".ogg") is None


def test_bank_format_and_info(tmp_path, monkeypatch):
    wav_root = tmp_path / "words"
    ogg_root = tmp_path / "words_ogg"
    rel = "en-us/gemini-Kore/0/1_cat.wav"       # path as stored in the manifest
    wav = wav_root / rel
    wav.parent.mkdir(parents=True)
    write_wav(wav, b"\x01\x00" * 4800)
    ogg = ogg_root / "en-us/gemini-Kore/0/1_cat.ogg"
    ogg.parent.mkdir(parents=True)
    ogg.write_bytes(b"OggS" + b"\x00" * 200)    # stand-in OGG copy

    m = Manifest(str(tmp_path / "wb.db"))
    m.add_words(["cat", "dog"], "en-us")
    ids = {r["word"]: r["id"] for r in m.iter_rows()}
    m.set_status(ids["cat"], DONE, file_path=rel)
    m.set_status(ids["dog"], DONE, file_path="en-us/gemini-Kore/0/2_dog.wav")  # files absent
    m.close()

    monkeypatch.setenv("BANK_DB", str(tmp_path / "wb.db"))
    monkeypatch.setenv("WAV_ROOT", str(wav_root))
    monkeypatch.setenv("OGG_ROOT", str(ogg_root))
    from src.server import app
    with TestClient(app) as c:
        info = c.get("/v1/bank/cat/info").json()
        assert info["wav"]["exists"] and info["wav"]["bytes"] == wav.stat().st_size
        assert info["ogg"]["exists"] and info["ogg"]["bytes"] == ogg.stat().st_size

        assert c.get("/v1/bank/cat", params={"format": "wav"}).content[:4] == b"RIFF"
        r_ogg = c.get("/v1/bank/cat", params={"format": "ogg"})
        assert r_ogg.status_code == 200 and r_ogg.content[:4] == b"OggS"

        # dog: in bank but neither file on disk
        dog = c.get("/v1/bank/dog/info").json()
        assert not dog["wav"]["exists"] and not dog["ogg"]["exists"]
        assert c.get("/v1/bank/dog", params={"format": "ogg"}).status_code == 404
    m.close()
