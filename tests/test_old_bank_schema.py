"""BankLookup must serve banks created before phrases grew voice/seed/ref_* columns.

Manifest ALTER-migrates old DBs when it opens them, but the serving path is
read-only and must not depend on that: a reader on an un-migrated bank used to
raise sqlite3.OperationalError ("no such column: seed") from get_phrase — a 500
on the real 2025 word bank that fresh-schema test fixtures could never catch.
"""
import sqlite3

from fastapi.testclient import TestClient

from wordbank.lookup import BankLookup
from wordbank.manifest import Manifest

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
    assert bank.get_phrase("good morning", "en-us", voice="Kore") is None
    bank.close()


def test_search_on_old_schema_bank(tmp_path):
    db = str(tmp_path / "old.db")
    _make_old_bank(db)
    bank = BankLookup(db)
    results = bank.search("good", "en-us")
    assert [r["text"] for r in results] == ["good morning"]
    assert results[0]["type"] == "phrase" and results[0]["voice"] is None
    bank.close()


def test_manifest_migrates_old_phrase_key_so_a_second_voice_fits(tmp_path):
    """Old phrases tables have UNIQUE(text, lang); ALTER adds `voice` but can't
    widen the key, so the same text in another voice used to be silently dropped."""
    db = str(tmp_path / "old.db")
    _make_old_bank(db)
    m = Manifest(db, table="phrases", text_col="text")
    assert m.add_words(["good morning"], "en-us", voice="Puck") == 1
    assert m.add_words(["good morning"], "en-us", voice="Puck") == 0   # still deduped
    m.close()

    m = Manifest(db, table="phrases", text_col="text")    # reopen: no second rebuild
    rows = {r["voice"]: r for r in m.iter_rows()}
    assert set(rows) == {None, "Puck"}
    assert rows[None]["status"] == "done"                 # original row kept intact
    assert rows[None]["file_path"] == "en/local/1/1_good_morning.wav"
    m.close()


def _phrase_bank(db, rows):
    """A current-schema bank holding `rows` (id, text, lang, voice, status,
    file_path) as written by older code: Manifest relabels on its next open."""
    Manifest(db).close()
    conn = sqlite3.connect(db)
    conn.executemany("INSERT INTO phrases (id, text, lang, voice, status, file_path) "
                     "VALUES (?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()


def _phrase_labels(db):
    m = Manifest(db, table="phrases", text_col="text")
    try:
        return {r["id"]: (r["voice"], r["file_path"]) for r in m.iter_rows()}
    finally:
        m.close()


def test_manifest_relabels_phrases_with_their_voices_word_label(tmp_path):
    """Phrases were "gemini-<voice>" while words were "<voice>"; a phrase now
    takes the word label: rewritten once on open, the clips (paths) untouched;
    any other label is left as is."""
    db = str(tmp_path / "wb.db")
    _phrase_bank(db, [
        (1, "good morning", "en-us", "gemini-Kore", "done", "en-us/gemini-Kore/phrases/0/1_a.wav"),
        (2, "good night", "en-us", "gemini-puck", "done", "en-us/gemini-puck/phrases/0/2_b.wav"),
        (3, "hello there", "en-gb", "gemini-en-gb-tutor-11", "done", "en-gb/gemini-en-gb-tutor-11/x.wav"),
        (4, "cheerio", "en-gb", "local-en-gb", "done", "en-gb/local-en-gb/0/4_c.wav"),
    ])
    labels = _phrase_labels(db)
    assert {i: v for i, (v, _) in labels.items()} == \
        {1: "Kore", 2: "Puck", 3: "en-gb-tutor-11", 4: "local-en-gb"}
    assert labels[2][1] == "en-us/gemini-puck/phrases/0/2_b.wav"
    assert _phrase_labels(db) == labels                     # reopen: nothing left to do


def test_phrase_relabel_collision_keeps_the_clip_then_the_oldest(tmp_path):
    """A relabel can meet a row already holding (text, lang, label). The one
    with a clip, then the lower id, takes the label; the other is deleted if
    it has no clip, else kept under a legacy spelling, where lookups still
    match it but serve the winner first."""
    db = str(tmp_path / "wb.db")
    _phrase_bank(db, [
        (1, "bye", "en-us", "Kore", "pending", None),           # no clip: deleted
        (2, "bye", "en-us", "gemini-Kore", "done", "a.wav"),    # takes the label
        (3, "hi", "en-us", "gemini-Kore", "done", "b.wav"),     # older clip: takes the label
        (4, "hi", "en-us", "Kore", "done", "c.wav"),            # newer: takes 3's old spelling
        (5, "yes", "en-us", "Kore", "done", "d.wav"),           # older clip: keeps the label
        (6, "yes", "en-us", "gemini-Kore", "failed", None),     # no clip: deleted
        (7, "yes", "en-us", "gemini-kore", "done", "e.wav"),    # newer: kept as it is
    ])
    labels = _phrase_labels(db)
    assert {i: v for i, (v, _) in labels.items()} == \
        {2: "Kore", 3: "Kore", 4: "gemini-Kore", 5: "Kore", 7: "gemini-kore"}
    assert _phrase_labels(db) == labels                     # reopen: the pairs stay

    bank = BankLookup(db)
    assert [bank.get_phrase(t, "en-us", voice="Kore")["file_path"] for t in ("bye", "hi", "yes")] \
        == ["a.wav", "b.wav", "d.wav"]
    assert bank.search("yes", "en-us") == [{"text": "yes", "type": "phrase", "voice": "Kore"}]
    bank.close()


def test_lookups_read_an_unmigrated_phrase_label_as_the_plain_one(client):
    """BankLookup never migrates (a freshly unpacked old archive may not have
    met a Manifest yet): a "gemini-Kore" phrase is served, pinned and
    reported as "Kore", the same as after the migration."""
    db = str(client.paths.db)
    _phrase_bank(db, [(1, "hello there", "en-us", "gemini-Kore", "done",
                       "en-us/gemini-Kore/phrases/0/1_hello-there.wav")])
    ogg = client.paths.ogg / "en-us/gemini-Kore/phrases/0/1_hello-there.ogg"
    ogg.parent.mkdir(parents=True)
    ogg.write_bytes(b"OggS-kore")

    for data in ({"voice": "kore"}, {}):
        r = client.post("/v1/speak", data={"text": "hello there", **data})
        assert (r.headers["x-source"], r.headers["x-voice"], r.content) == ("bank", "Kore", b"OggS-kore")
    assert client.get("/v1/search", params={"prefix": "hello", "voice": "Kore"}).json()["results"] \
        == [{"text": "hello there", "type": "phrase", "voice": "Kore"}]
    for pin in ("Kore", "gemini-Kore"):                     # the legacy spelling too
        info = client.get("/v1/phrase/info", params={"text": "hello there", "voice": pin})
        assert info.status_code == 200 and info.json()["voice"] == "Kore", pin
    assert client.gemini.calls == []
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT voice FROM phrases").fetchall() == [("gemini-Kore",)]   # not migrated
    conn.close()


def test_speak_into_old_schema_bank(tmp_path, bank_env, fake_gemini):
    """End to end: /v1/speak on the real (old-schema) bank, for a phrase that
    already exists without a voice, generates, saves, and then serves it."""
    _make_old_bank(str(bank_env.db))
    from src.server import app
    with TestClient(app) as c:
        miss = c.post("/v1/speak", data={"text": "good morning", "voice": "Puck"})
        assert miss.status_code == 200, miss.text
        hit = c.post("/v1/speak", data={"text": "good morning", "voice": "Puck"})
        assert hit.headers["x-source"] == "bank" and hit.content == miss.content


def test_manifest_adds_provenance_columns_to_an_old_bank(tmp_path):
    """model/style are ALTER-added to both tables; existing rows stay NULL.
    A generation result (model given) records both, style exactly as given;
    a status-only update keeps them."""
    db = str(tmp_path / "old.db")
    _make_old_bank(db)
    for table, col in (("words", "word"), ("phrases", "text")):
        m = Manifest(db, table=table, text_col=col)
        [old] = m.iter_rows()
        assert (old["model"], old["style"], old["status"]) == (None, None, "done")

        m.set_status(old["id"], "done", file_path=old["file_path"],
                     model="gemini-3.8-flash-tts", style="plain")
        m.set_status(old["id"], "failed", error="later retry")          # status only
        [row] = m.iter_rows()
        assert (row["model"], row["style"]) == ("gemini-3.8-flash-tts", "plain")

        m.set_status(old["id"], "done", file_path=old["file_path"],
                     model="gemini-3.8-flash-lite-tts")                   # re-take, no style
        [row] = m.iter_rows()
        assert (row["model"], row["style"]) == ("gemini-3.8-flash-lite-tts", None)
        m.close()

    bank = BankLookup(db)                       # the reader still serves it
    assert bank.get_phrase("good morning", "en-us") is not None
    bank.close()


# The words table as it was before voice joined its key (and before QC).
_PRE_VOICE_KEY_WORDS = """
CREATE TABLE words (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word TEXT NOT NULL, lang TEXT NOT NULL, voice TEXT,
    status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
    file_path TEXT, transcript TEXT, billed_chars INTEGER, seed INTEGER,
    ref_audio TEXT, ref_text TEXT, model TEXT, style TEXT, error TEXT,
    batch_id TEXT, created_at REAL, updated_at REAL,
    UNIQUE(word, lang)
);
CREATE INDEX idx_words_lang_word ON words(lang, word);
INSERT INTO words (id, word, lang, voice, status, file_path, error) VALUES
  (1, 'cat', 'en-gb', 'en-gb-tutor-11', 'done', 'en-gb/gemini-en-gb-tutor-11/0/1_cat.wav',
   'review: ends on sound (-31 dB)'),
  (2, 'dog', 'en-us', NULL, 'done', 'en/gemini-Kore/0/2_dog.wav', NULL),
  (3, 'emu', 'en-gb', NULL, 'submitted', NULL, NULL),
  (4, 'gnu', 'fr-fr', NULL, 'pending', NULL, NULL),
  (5, 'yak', 'fr-fr', NULL, 'pending', NULL, NULL);
DELETE FROM words WHERE id = 5;
"""


def test_manifest_migrates_the_old_word_key_keeping_rows_and_ids(tmp_path):
    """UNIQUE(word, lang) held one clip per word: rebuilt once with voice in
    the key, every row and id kept, the AUTOINCREMENT mark too (ids name
    files). A NULL voice is recovered from the clip's path; a row not
    generated yet takes its language's only voice, else stays unattributed
    and is claimed by the first voice to generate it."""
    db = str(tmp_path / "old.db")
    conn = sqlite3.connect(db)
    conn.executescript(_PRE_VOICE_KEY_WORDS)
    conn.close()

    m = Manifest(db)
    rows = {r["id"]: r for r in m.iter_rows()}
    assert [(r["word"], r["voice"]) for r in rows.values()] == [
        ("cat", "en-gb-tutor-11"), ("dog", "Kore"), ("emu", "en-gb-tutor-11"), ("gnu", None)]
    assert (rows[1]["review"], rows[1]["error"], rows[1]["qc_end_db"]) == \
        ("ends on sound (-31 dB)", None, None)                 # flag.py's note moved
    before = [dict(r) for r in m.iter_rows()]

    assert m.add_words(["cat", "emu"], "en-gb", voice="en-gb-tutor-14") == 2
    assert m.get_entry("cat", "en-gb", "en-gb-tutor-14")["id"] == 6   # 5 is never reused
    assert m.add_words(["gnu"], "fr-fr", voice="fr-fr-x") == 0        # claimed by generating
    assert [r["word"] for r in m.pending(lang="fr-fr", voice="fr-fr-x")] == ["gnu"]
    # voice-less, as code predating the key adds: still one row per word
    assert m.add_words(["cat"], "en-gb") == m.add_words(["dog"], "en-us") == 0
    m.close()

    m = Manifest(db)                           # reopen: no second migration
    assert [dict(r) for r in m.iter_rows()][:4] == before
    assert len(m.iter_rows()) == 6
    m.close()
    bank = BankLookup(db)
    assert bank.get("dog", "en-us", voice="Kore")["file_path"] == "en/gemini-Kore/0/2_dog.wav"
    bank.close()
