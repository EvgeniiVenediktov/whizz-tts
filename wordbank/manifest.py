"""SQLite manifest tracking per-word generation status. Enables resume + dedup.

Thread-safe: a single connection (check_same_thread=False) guarded by a lock.
DB writes are tiny; the real bottleneck is the network TTS call, so lock
contention is negligible even with many worker threads.
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
    status       TEXT NOT NULL DEFAULT 'pending',
    attempts     INTEGER NOT NULL DEFAULT 0,
    file_path    TEXT,
    transcript   TEXT,
    billed_chars INTEGER,
    error        TEXT,
    batch_id     TEXT,
    created_at   REAL,
    updated_at   REAL,
    UNIQUE(word, lang)
);
-- (lang, word) leading index → per-language exact lookup AND prefix range scan
-- seek straight into one language; also serves `GROUP BY lang` for /v1/languages.
CREATE INDEX IF NOT EXISTS idx_words_lang_word ON words(lang, word);

-- Voice-clone references: one row per reusable reference clip. The local
-- OmniVoice path clones `ref_audio` (+ `ref_text`) to speak a whole bank in
-- that voice/accent. ref_audio is a file path (e.g. refs/en-gb/ref.wav); the
-- bank's served audio lives in `words` — this table is generation-side only.
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
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        # Forward-compat: add batch_id to manifests created before it existed.
        try:
            self._conn.execute("ALTER TABLE words ADD COLUMN batch_id TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def add_words(self, words: Iterable[str], lang: str) -> int:
        """Insert words (deduped by (word, lang)). Returns count newly added."""
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            before = cur.execute("SELECT COUNT(*) FROM words").fetchone()[0]
            cur.executemany(
                "INSERT OR IGNORE INTO words (word, lang, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                [(w, lang, now, now) for w in words],
            )
            self._conn.commit()
            after = cur.execute("SELECT COUNT(*) FROM words").fetchone()[0]
        return after - before

    def pending(self, limit: Optional[int] = None, max_attempts: int = 3):
        sql = (
            "SELECT * FROM words WHERE status IN (?, ?) AND attempts < ? "
            "ORDER BY id"
        )
        params = [PENDING, FAILED, max_attempts]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def start_attempt(self, word_id: int) -> int:
        """Mark a word in-progress, increment attempts, return the new count."""
        with self._lock:
            self._conn.execute(
                "UPDATE words SET attempts = attempts + 1, status = ?, "
                "updated_at = ? WHERE id = ?",
                (PENDING, time.time(), word_id),
            )
            self._conn.commit()
            return self._conn.execute(
                "SELECT attempts FROM words WHERE id = ?", (word_id,)
            ).fetchone()[0]

    def set_status(self, word_id: int, status: str, *, error: Optional[str] = None,
                   file_path: Optional[str] = None, transcript: Optional[str] = None,
                   billed_chars: Optional[int] = None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE words SET status = ?, error = ?, file_path = ?, "
                "transcript = ?, billed_chars = ?, updated_at = ? WHERE id = ?",
                (status, error, file_path, transcript, billed_chars,
                 time.time(), word_id),
            )
            self._conn.commit()

    def mark_submitted(self, word_ids, batch_id: str) -> None:
        """Mark words as enqueued in an async batch job (counts as one attempt)."""
        now = time.time()
        with self._lock:
            self._conn.executemany(
                "UPDATE words SET status = ?, batch_id = ?, attempts = attempts + 1, "
                "updated_at = ? WHERE id = ?",
                [(SUBMITTED, batch_id, now, wid) for wid in word_ids],
            )
            self._conn.commit()

    def get_by_batch(self, batch_id: str):
        """Rows in a batch, in submission (id) order — aligns with inlined_responses."""
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM words WHERE batch_id = ? ORDER BY id", (batch_id,)
            ).fetchall()

    def active_batch_ids(self):
        """Batch ids with words still awaiting results (for resume)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT batch_id FROM words "
                "WHERE status = ? AND batch_id IS NOT NULL",
                (SUBMITTED,),
            ).fetchall()
        return [r[0] for r in rows]

    def counts(self) -> dict:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) FROM words GROUP BY status"
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def iter_rows(self):
        with self._lock:
            return self._conn.execute("SELECT * FROM words ORDER BY id").fetchall()

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
