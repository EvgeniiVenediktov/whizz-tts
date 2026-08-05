"""BankLookup must serve banks created before phrases grew voice/seed/ref_* columns.

Manifest ALTER-migrates old DBs when it opens them, but the serving path is
read-only and must not depend on that: a reader on an un-migrated bank used to
raise sqlite3.OperationalError ("no such column: seed") from get_phrase — a 500
on the real 2025 word bank that fresh-schema test fixtures could never catch.
"""
import sqlite3

from wordbank.lookup import BankLookup

_OLD_SCHEMA = """
CREATE TABLE words (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word TEXT NOT NULL, lang TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    file_path TEXT, transcript TEXT, billed_chars INTEGER, error TEXT,
    created_at REAL, updated_at REAL,
    UNIQUE(word, lang)
);
CREATE TABLE phrases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL, lang TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    file_path TEXT, transcript TEXT, billed_chars INTEGER, error TEXT,
    created_at REAL, updated_at REAL,
    UNIQUE(text, lang)
);
"""


def _make_old_bank(path):
    conn = sqlite3.connect(path)
    conn.executescript(_OLD_SCHEMA)
    conn.execute(
        "INSERT INTO words (word, lang, status, file_path) VALUES (?, ?, ?, ?)",
        ("hello", "en-us", "done", "en/1/1_hello.wav"),
    )
    conn.execute(
        "INSERT INTO phrases (text, lang, status, file_path) VALUES (?, ?, ?, ?)",
        ("good morning", "en-us", "done", "en/local/1/1_good_morning.wav"),
    )
    conn.commit()
    conn.close()


def test_get_phrase_on_old_schema_bank(tmp_path):
    db = str(tmp_path / "old.db")
    _make_old_bank(db)
    bank = BankLookup(db)
    row = bank.get_phrase("good morning", "en-us")
    assert row is not None
    assert row["file_path"] == "en/local/1/1_good_morning.wav"
    assert row["voice"] is None and row["seed"] is None
    assert row["ref_audio"] is None and row["ref_text"] is None
    bank.close()


def test_voice_pinned_lookup_on_voiceless_bank_is_a_miss(tmp_path):
    db = str(tmp_path / "old.db")
    _make_old_bank(db)
    bank = BankLookup(db)
    assert bank.get_phrase("good morning", "en-us", voice="gemini-Kore") is None
    bank.close()


def test_search_on_old_schema_bank(tmp_path):
    db = str(tmp_path / "old.db")
    _make_old_bank(db)
    bank = BankLookup(db)
    results = bank.search("good", "en-us")
    assert [r["text"] for r in results] == ["good morning"]
    assert results[0]["type"] == "phrase" and results[0]["voice"] is None
    bank.close()
