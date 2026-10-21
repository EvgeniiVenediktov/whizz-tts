"""Tests for the `voices` table (clone-reference registry) on the manifest,
the storage/resolution helpers, and read-only lookup access."""
import os

from wordbank.lookup import BankLookup
from wordbank.manifest import Manifest
from wordbank.voices import resolve_voice_path, save_voice_clip, voice_rel_path


def test_add_and_get_voice(tmp_path):
    m = Manifest(str(tmp_path / "wb.db"))
    m.add_voice("en-gb", "en-gb", "refs/en-gb/ref.wav",
                ref_text="The quick brown fox.", instruct="British accent",
                source="gemini", sample_rate=24_000)

    v = m.get_voice("en-gb", "en-gb")
    assert v is not None
    assert v["ref_audio"] == "refs/en-gb/ref.wav"
    assert v["ref_text"] == "The quick brown fox."
    assert v["instruct"] == "British accent"
    assert v["source"] == "gemini"
    assert v["sample_rate"] == 24_000
    m.close()


def test_get_voice_missing_returns_none(tmp_path):
    m = Manifest(str(tmp_path / "wb.db"))
    assert m.get_voice("nope") is None
    assert m.get_voice("nope", "en-gb") is None
    m.close()


def test_add_voice_upserts_on_name_lang(tmp_path):
    m = Manifest(str(tmp_path / "wb.db"))
    m.add_voice("en-gb", "en-gb", "refs/old.wav", ref_text="old")
    m.add_voice("en-gb", "en-gb", "refs/new.wav", ref_text="new")  # same key → update

    assert len(m.voices()) == 1
    v = m.get_voice("en-gb", "en-gb")
    assert v["ref_audio"] == "refs/new.wav" and v["ref_text"] == "new"
    m.close()


def test_same_name_different_lang_coexist(tmp_path):
    m = Manifest(str(tmp_path / "wb.db"))
    m.add_voice("narrator", "en-us", "refs/us.wav")
    m.add_voice("narrator", "en-gb", "refs/gb.wav")

    assert len(m.voices()) == 2
    assert m.get_voice("narrator", "en-us")["ref_audio"] == "refs/us.wav"
    assert m.get_voice("narrator", "en-gb")["ref_audio"] == "refs/gb.wav"
    # name-only lookup returns the first by id (deterministic)
    assert m.get_voice("narrator")["ref_audio"] == "refs/us.wav"
    m.close()


# ---- storage + resolution helpers ----

def test_save_voice_clip_writes_file_and_registers(tmp_path):
    m = Manifest(str(tmp_path / "wb.db"))
    voices_root = tmp_path / "voices"
    rel = save_voice_clip(m, str(voices_root), "kore", "en-us", b"RIFFwavebytes",
                          "hi there", instruct="warm", source="gemini")

    assert rel == voice_rel_path("en-us", "kore") == os.path.join("en-us", "kore.wav")
    assert (voices_root / rel).read_bytes() == b"RIFFwavebytes"
    v = m.get_voice("kore", "en-us")
    assert v["ref_audio"] == rel and v["ref_text"] == "hi there"
    assert v["instruct"] == "warm" and v["source"] == "gemini"
    m.close()


def test_resolve_voice_path_rules(tmp_path):
    # relative + not on disk → under voices root
    assert resolve_voice_path("/voices", "en-us/kore.wav") == "/voices/en-us/kore.wav"
    # absolute → passes through untouched
    assert resolve_voice_path("/voices", "/abs/ref.wav") == "/abs/ref.wav"
    # relative but exists from CWD (CLI-registered repo path) → as-given
    f = tmp_path / "ref.wav"
    f.write_bytes(b"x")
    assert resolve_voice_path("/voices", str(f)) == str(f)


# ---- read-only lookup access ----

def test_lookup_reads_voices(tmp_path):
    db = str(tmp_path / "wb.db")
    m = Manifest(db)
    m.add_voice("kore", "en-us", "en-us/kore.wav", ref_text="hi", source="gemini")
    m.close()

    bl = BankLookup(db)
    assert bl.voices() == [{
        "name": "kore", "lang": "en-us", "ref_audio": "en-us/kore.wav",
        "ref_text": "hi", "instruct": None, "source": "gemini",
    }]
    assert bl.voices("en-gb") == []
    assert bl.get_voice("kore", "en-us")["ref_text"] == "hi"
    assert bl.get_voice("nope", "en-us") is None
    bl.close()


def test_lookup_voices_empty_when_no_table(tmp_path):
    """A bank DB with no `voices` table (older bank) reads as empty, not an error."""
    import sqlite3
    db = str(tmp_path / "old.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE words (id INTEGER PRIMARY KEY, word TEXT, lang TEXT, status TEXT)")
    conn.commit()
    conn.close()
    bl = BankLookup(db)
    assert bl.voices() == []
    assert bl.get_voice("x", "en-us") is None
    bl.close()
