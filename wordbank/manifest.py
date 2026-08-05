"""SQLite manifest tracking per-entry generation status. Enables resume + dedup.

Two content tables with identical shape:
  - `words`   — the Gemini single-word bank   (UNIQUE(word, lang))
  - `phrases` — the local OmniVoice phrase bank (UNIQUE(text, lang, voice))
They are stored separately so words and phrases never overlap. One `Manifest`
instance targets one table (pass `table=`/`text_col=`); both tables (and the
shared `voices` reference registry) are always created.

Thread-safe: a single connection (check_same_thread=False) guarded by a lock.
DB writes are tiny; the real bottleneck is the TTS call, so lock contention is
negligible even with many worker threads.
"""
import sqlite3
import threading
import time
from typing import Iterable, Optional

PENDING = "pending"
SUBMITTED = "submitted"   # enqueued in an async batch job, awaiting results
DONE = "done"            # audio written to disk (batch path; not STT-verified)
VERIFIED = "verified"    # audio written AND confirmed by STT (sync path)
FAILED = "failed"
QUARANTINED = "quarantined"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS words (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    word         TEXT NOT NULL,
    lang         TEXT NOT NULL,
    voice        TEXT,
    status       TEXT NOT NULL DEFAULT 'pending',
    attempts     INTEGER NOT NULL DEFAULT 0,
    file_path    TEXT,
    transcript   TEXT,
    billed_chars INTEGER,
    seed         INTEGER,
    ref_audio    TEXT,
    ref_text     TEXT,
    error        TEXT,
    batch_id     TEXT,
    created_at   REAL,
    updated_at   REAL,
    UNIQUE(word, lang)
);
-- (lang, word) leading index → per-language exact lookup AND prefix range scan
-- seek straight into one language; also serves `GROUP BY lang` for /v1/languages.
CREATE INDEX IF NOT EXISTS idx_words_lang_word ON words(lang, word);

-- Phrases: the local OmniVoice phrase bank. Separate from `words` so the two
-- never collide. A phrase may exist in several voices, hence voice is in the key.
CREATE TABLE IF NOT EXISTS phrases (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    text         TEXT NOT NULL,
    lang         TEXT NOT NULL,
    voice        TEXT,
    status       TEXT NOT NULL DEFAULT 'pending',
    attempts     INTEGER NOT NULL DEFAULT 0,
    file_path    TEXT,
    transcript   TEXT,
    billed_chars INTEGER,
    seed         INTEGER,
    ref_audio    TEXT,
    ref_text     TEXT,
    error        TEXT,
    batch_id     TEXT,
    created_at   REAL,
    updated_at   REAL,
    UNIQUE(text, lang, voice)
);
CREATE INDEX IF NOT EXISTS idx_phrases_lang_text ON phrases(lang, text);

-- Voice-clone references: one row per reusable reference clip. The local
-- OmniVoice path clones `ref_audio` (+ `ref_text`) to speak a whole bank in
-- that voice/accent. ref_audio is a file path (e.g. refs/en-gb/ref.wav).
CREATE TABLE IF NOT EXISTS voices (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    lang         TEXT NOT NULL,
    ref_audio    TEXT NOT NULL,
    ref_text     TEXT,
    instruct     TEXT,
    source       TEXT,
    sample_rate  INTEGER,
    created_at   REAL,
    updated_at   REAL,
    UNIQUE(name, lang)
);
CREATE INDEX IF NOT EXISTS idx_voices_lang ON voices(lang);
"""


class Manifest:
    def __init__(self, path: str, *, table: str = "words", text_col: str = "word"):
        self.path = path
        self.table = table
        self.text_col = text_col
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        # Forward-compat: add columns to manifests created before they existed.
        for tbl in ("words", "phrases"):
            for col, decl in (("batch_id", "TEXT"), ("voice", "TEXT"),
                              ("seed", "INTEGER"), ("ref_audio", "TEXT"),
                              ("ref_text", "TEXT")):
                try:
                    self._conn.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {decl}")
                except sqlite3.OperationalError:
                    pass  # column already exists

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def add_words(self, words: Iterable[str], lang: str,
                  voice: Optional[str] = None) -> int:
        """Insert entries (deduped by the table's UNIQUE key). Returns count added.

        `voice` is part of the key for phrases (so the same phrase can be added
        in multiple voices); for words it is just recorded.
        """
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            before = cur.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0]
            cur.executemany(
                f"INSERT OR IGNORE INTO {self.table} "
                f"({self.text_col}, lang, voice, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                [(w, lang, voice, now, now) for w in words],
            )
            self._conn.commit()
            after = cur.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0]
        return after - before

    def pending(self, limit: Optional[int] = None, max_attempts: int = 3):
        sql = (
            f"SELECT * FROM {self.table} WHERE status IN (?, ?) AND attempts < ? "
            "ORDER BY id"
        )
        params = [PENDING, FAILED, max_attempts]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def start_attempt(self, row_id: int) -> int:
        """Mark an entry in-progress, increment attempts, return the new count."""
        with self._lock:
            self._conn.execute(
                f"UPDATE {self.table} SET attempts = attempts + 1, status = ?, "
                "updated_at = ? WHERE id = ?",
                (PENDING, time.time(), row_id),
            )
            self._conn.commit()
            return self._conn.execute(
                f"SELECT attempts FROM {self.table} WHERE id = ?", (row_id,)
            ).fetchone()[0]

    def set_status(self, row_id: int, status: str, *, error: Optional[str] = None,
                   file_path: Optional[str] = None, transcript: Optional[str] = None,
                   billed_chars: Optional[int] = None, voice: Optional[str] = None,
                   seed: Optional[int] = None, ref_audio: Optional[str] = None,
                   ref_text: Optional[str] = None) -> None:
        # voice/seed/ref_* use COALESCE so a later status-only update (e.g. a
        # retry marking FAILED) never clobbers recorded provenance.
        with self._lock:
            self._conn.execute(
                f"UPDATE {self.table} SET status = ?, error = ?, file_path = ?, "
                "transcript = ?, billed_chars = ?, voice = COALESCE(?, voice), "
                "seed = COALESCE(?, seed), ref_audio = COALESCE(?, ref_audio), "
                "ref_text = COALESCE(?, ref_text), updated_at = ? WHERE id = ?",
                (status, error, file_path, transcript, billed_chars, voice,
                 seed, ref_audio, ref_text, time.time(), row_id),
            )
            self._conn.commit()

    def mark_submitted(self, row_ids, batch_id: str) -> None:
        """Mark entries as enqueued in an async batch job (counts as one attempt)."""
        now = time.time()
        with self._lock:
            self._conn.executemany(
                f"UPDATE {self.table} SET status = ?, batch_id = ?, "
                "attempts = attempts + 1, updated_at = ? WHERE id = ?",
                [(SUBMITTED, batch_id, now, wid) for wid in row_ids],
            )
            self._conn.commit()

    def get_by_batch(self, batch_id: str):
        """Rows in a batch, in submission (id) order — aligns with inlined_responses."""
        with self._lock:
            return self._conn.execute(
                f"SELECT * FROM {self.table} WHERE batch_id = ? ORDER BY id",
                (batch_id,),
            ).fetchall()

    def active_batch_ids(self):
        """Batch ids with entries still awaiting results (for resume)."""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT DISTINCT batch_id FROM {self.table} "
                "WHERE status = ? AND batch_id IS NOT NULL",
                (SUBMITTED,),
            ).fetchall()
        return [r[0] for r in rows]

    def counts(self) -> dict:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT status, COUNT(*) FROM {self.table} GROUP BY status"
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def iter_rows(self):
        with self._lock:
            return self._conn.execute(
                f"SELECT * FROM {self.table} ORDER BY id"
            ).fetchall()

    def get_entry(self, text: str, lang: str, voice: Optional[str] = None):
        """Fetch one row from this manifest's table by (text_col, lang), optionally
        pinned to a voice. Returns the row or None. Unlike BankLookup.get*, this is
        not status-scoped (used by writers to find a row they just inserted)."""
        sql = f"SELECT * FROM {self.table} WHERE {self.text_col} = ? AND lang = ?"
        params = [text, lang]
        if voice is not None:
            sql += " AND voice = ?"
            params.append(voice)
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    # ---- voices (clone references) ----

    def add_voice(self, name: str, lang: str, ref_audio: str, *,
                  ref_text: Optional[str] = None, instruct: Optional[str] = None,
                  source: Optional[str] = None,
                  sample_rate: Optional[int] = None) -> None:
        """Register (or update) a voice-clone reference. Upsert on (name, lang)."""
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO voices (name, lang, ref_audio, ref_text, instruct, "
                "source, sample_rate, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(name, lang) DO UPDATE SET "
                "ref_audio=excluded.ref_audio, ref_text=excluded.ref_text, "
                "instruct=excluded.instruct, source=excluded.source, "
                "sample_rate=excluded.sample_rate, updated_at=excluded.updated_at",
                (name, lang, ref_audio, ref_text, instruct, source, sample_rate,
                 now, now),
            )
            self._conn.commit()

    def get_voice(self, name: str, lang: Optional[str] = None):
        """Fetch one registered voice by name (optionally scoped to lang). None if absent."""
        with self._lock:
            if lang is not None:
                return self._conn.execute(
                    "SELECT * FROM voices WHERE name = ? AND lang = ?", (name, lang)
                ).fetchone()
            return self._conn.execute(
                "SELECT * FROM voices WHERE name = ? ORDER BY id LIMIT 1", (name,)
            ).fetchone()

    def voices(self):
        """All registered voices, ordered by (lang, name)."""
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM voices ORDER BY lang, name"
            ).fetchall()
