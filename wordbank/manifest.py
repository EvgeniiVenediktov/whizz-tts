"""SQLite manifest of generation status (resume + dedup), two tables of one
shape: `words` (UNIQUE(lang, word, voice)) and `phrases`, the /v1/speak bank
(UNIQUE(text, lang, voice)), kept apart but sharing voice labels ("Kore").
A Manifest targets one (`table=`/`text_col=`); both are always created.

A NULL word `voice` is unattributed (ingested by code before the voice key,
not generated yet): the first voice to generate it claims it, and the
`words_add_dedup` trigger keeps adding that word from creating a second row.

Thread-safe: one connection (check_same_thread=False) behind a lock.
"""
import sqlite3
import threading
import time
import uuid
from typing import Iterable, Optional

from .paths import voice_from_path

PENDING = "pending"
SUBMITTED = "submitted"   # enqueued in an async batch job, awaiting results
DONE = "done"            # audio written to disk
FAILED = "failed"
# A batch take QC flagged, queued for a re-take (best take at qc.retake_path).
# Not served; re-taken only with the row's own model and direction.
RETAKE = "retake"

# SET clause for the generation provenance (params: model, model, style): with a
# model, both columns take the given values (style None = "no direction");
# without one, both are kept. See Manifest.set_status.
_PROVENANCE_SQL = ("model = COALESCE(?, model), "
                   "style = CASE WHEN ? IS NULL THEN style ELSE ? END")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS words (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    word         TEXT NOT NULL,
    lang         TEXT NOT NULL,
    voice        TEXT,
    status       TEXT NOT NULL DEFAULT 'pending',
    attempts     INTEGER NOT NULL DEFAULT 0,
    file_path    TEXT,
    billed_chars INTEGER,
    model        TEXT,
    style        TEXT,
    error        TEXT,
    review       TEXT,
    qc_seconds   REAL,
    qc_end_db    REAL,
    qc_takes     INTEGER NOT NULL DEFAULT 0,
    qc_accepted  INTEGER NOT NULL DEFAULT 0,
    batch_id     TEXT,
    created_at   REAL,
    updated_at   REAL,
    -- lang leading: the key's index is also the lookup index — per-language
    -- exact lookup and prefix range scan seek straight into one language
    -- (and one voice), and it serves `GROUP BY lang` for /v1/languages.
    UNIQUE(lang, word, voice)
);

-- Phrases: the /v1/speak phrase bank. Separate from `words` so the two
-- never collide. A phrase may exist in several voices, hence voice is in the key.
CREATE TABLE IF NOT EXISTS phrases (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    text         TEXT NOT NULL,
    lang         TEXT NOT NULL,
    voice        TEXT,
    status       TEXT NOT NULL DEFAULT 'pending',
    attempts     INTEGER NOT NULL DEFAULT 0,
    file_path    TEXT,
    billed_chars INTEGER,
    model        TEXT,
    style        TEXT,
    error        TEXT,
    review       TEXT,
    qc_seconds   REAL,
    qc_end_db    REAL,
    qc_takes     INTEGER NOT NULL DEFAULT 0,
    qc_accepted  INTEGER NOT NULL DEFAULT 0,
    batch_id     TEXT,
    clip_id      TEXT,      -- uuid4: /v1/clip/{id}, and the file name of a long text
    created_at   REAL,
    updated_at   REAL,
    UNIQUE(text, lang, voice)
);
"""

# After the key migrations (a rebuilt table loses its indexes and triggers).
_POST_SCHEMA = """
CREATE INDEX IF NOT EXISTS idx_phrases_lang_text ON phrases(lang, text);
CREATE UNIQUE INDEX IF NOT EXISTS idx_phrases_clip_id ON phrases(clip_id);

-- A word is added for a voice unless that voice has it or an unattributed row
-- still to generate covers it; a voice-less add is one row per word (NULL-
-- distinct UNIQUE can't say so). A trigger, so older writers obey it too.
CREATE TRIGGER IF NOT EXISTS words_add_dedup BEFORE INSERT ON words
WHEN EXISTS (SELECT 1 FROM words WHERE lang = NEW.lang AND word = NEW.word
             AND (NEW.voice IS NULL OR voice = NEW.voice
                  OR (voice IS NULL AND status != 'done')))
BEGIN SELECT RAISE(IGNORE); END;
"""

# QC columns (see qc.Check); NULL measurements = never checked. qc_accepted = 1:
# a person listened and kept the clip despite its flag, so `qc` skips it.
_QC_COLUMNS = (("review", "TEXT"), ("qc_seconds", "REAL"), ("qc_end_db", "REAL"),
               ("qc_takes", "INTEGER NOT NULL DEFAULT 0"),
               ("qc_accepted", "INTEGER NOT NULL DEFAULT 0"))


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
            for col, decl in (("batch_id", "TEXT"), ("voice", "TEXT"), ("model", "TEXT"),
                              ("style", "TEXT"), *_QC_COLUMNS):
                try:
                    self._conn.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {decl}")
                except sqlite3.OperationalError:
                    pass  # column already exists
        try:
            self._conn.execute("ALTER TABLE phrases ADD COLUMN clip_id TEXT")
        except sqlite3.OperationalError:
            pass
        # A key without `voice` silently dropped a second voice's row, and
        # ALTER can't widen a UNIQUE key: rebuilt once.
        if ["text", "lang"] in self._unique_keys("phrases"):
            self._rebuild("phrases")
        if ["lang", "word", "voice"] not in self._unique_keys("words"):
            self._rebuild("words", self._migrate_word_rows)
        self._conn.executemany("UPDATE phrases SET clip_id = ? WHERE id = ?", [
            (str(uuid.uuid4()), r[0])
            for r in self._conn.execute("SELECT id FROM phrases WHERE clip_id IS NULL")])
        self._conn.commit()
        self._conn.executescript(_POST_SCHEMA)

    def _unique_keys(self, table: str) -> list:
        """Column lists of the table's UNIQUE constraints."""
        return [
            [c[2] for c in self._conn.execute(f'PRAGMA index_info("{idx[1]}")')]
            for idx in self._conn.execute(f"PRAGMA index_list({table})")
            if idx[2] and idx[3] == "u"
        ]

    def _rebuild(self, table: str, migrate_rows=None) -> None:
        """Recreate `table` from _SCHEMA once (a UNIQUE key can't be ALTERed),
        keeping every row, column and id, and the AUTOINCREMENT high-water mark:
        file names embed the id, so an id is never handed out twice.
        `migrate_rows` runs on the new table in the same transaction."""
        old_cols = [(r[1], r[2]) for r in self._conn.execute(f"PRAGMA table_info({table})")]
        start = _SCHEMA.index(f"CREATE TABLE IF NOT EXISTS {table} ")
        create = _SCHEMA[start:_SCHEMA.index(");", start) + 2]
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                seq = self._conn.execute("SELECT seq FROM sqlite_sequence WHERE name = ?",
                                         (table,)).fetchone()
                self._conn.execute(f"ALTER TABLE {table} RENAME TO {table}_old")
                self._conn.execute(create)
                new_cols = {r[1] for r in self._conn.execute(f"PRAGMA table_info({table})")}
                for name, decl in old_cols:
                    if name not in new_cols:        # a column _SCHEMA doesn't know: kept
                        self._conn.execute(f'ALTER TABLE {table} ADD COLUMN "{name}" {decl}')
                cols = ", ".join(f'"{name}"' for name, _ in old_cols)
                self._conn.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM {table}_old")
                self._conn.execute(f"DROP TABLE {table}_old")
                if seq:
                    if not self._conn.execute(
                            "UPDATE sqlite_sequence SET seq = MAX(seq, ?) WHERE name = ?",
                            (seq[0], table)).rowcount:
                        self._conn.execute("INSERT INTO sqlite_sequence (name, seq) VALUES (?, ?)",
                                           (table, seq[0]))
                if migrate_rows is not None:
                    migrate_rows()
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def _migrate_word_rows(self) -> None:
        """One-time data moves of the words key migration (no collisions under
        the old UNIQUE(word, lang)): a clip's NULL voice is recovered from its
        file_path (`<lang>/gemini-<voice>/…`); a pending or failed row takes its
        language's voice when all its clips are in one (the voice its run was
        for). A row in flight keeps NULL: its batch may be another voice's."""
        for row_id, file_path in self._conn.execute(
                "SELECT id, file_path FROM words WHERE voice IS NULL "
                "AND file_path IS NOT NULL").fetchall():
            voice = voice_from_path(file_path)
            if voice:
                self._conn.execute("UPDATE words SET voice = ? WHERE id = ?", (voice, row_id))
        for lang, voice in self._conn.execute(
                "SELECT lang, MIN(voice) FROM words WHERE voice IS NOT NULL "
                "GROUP BY lang HAVING COUNT(DISTINCT voice) = 1").fetchall():
            self._conn.execute(       # not SUBMITTED: its batch's voice may be any
                "UPDATE words SET voice = ? WHERE lang = ? AND voice IS NULL "
                "AND status NOT IN (?, ?)", (voice, lang, DONE, SUBMITTED))

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def add_words(self, words: Iterable[str], lang: str,
                  voice: Optional[str] = None) -> int:
        """Insert entries (deduped by the table's UNIQUE key). Returns count added.

        `voice` is part of the key of both tables, so the same text can be
        added in several voices. For words, a word with an unattributed row
        still to generate isn't added again for a voice, and a voice-less add
        dedupes on (word, lang) alone (the `words_add_dedup` trigger).
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

    def pending(self, limit: Optional[int] = None, max_attempts: int = 3,
                lang: Optional[str] = None, voice: Optional[str] = None,
                retake: Optional[tuple] = None):
        """Rows still to generate (pending, or failed under max_attempts), in id
        order. `lang` restricts them to one language. `voice` to one voice's
        rows, plus the unattributed ones (voice NULL) that voice has no row
        for: the first voice to generate such a row claims it. `retake=(model,
        style)` adds the RETAKE rows recorded with exactly that provenance, so
        a QC re-take uses the take's own model and direction."""
        t, col = self.table, self.text_col
        sql = f"SELECT * FROM {t} WHERE ((status IN (?, ?) AND attempts < ?)"
        params = [PENDING, FAILED, max_attempts]
        if retake is not None:
            sql += " OR (status = ? AND model IS ? AND style IS ?)"
            params += [RETAKE, *retake]
        sql += ") "
        if lang is not None:
            sql += "AND lang = ? "
            params.append(lang)
        if voice is not None:
            sql += (f"AND (voice = ? OR (voice IS NULL AND NOT EXISTS (SELECT 1 FROM {t} o "
                    f"WHERE o.lang = {t}.lang AND o.{col} = {t}.{col} AND o.voice = ?))) ")
            params += [voice, voice]
        sql += "ORDER BY id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def claim(self, row_id: int, voice: str) -> bool:
        """Give an unattributed row `voice` before generating it; False when
        another voice's run claimed it first."""
        with self._lock:
            n = self._conn.execute(f"UPDATE {self.table} SET voice = ? WHERE id = ? "
                                   "AND (voice IS NULL OR voice = ?)",
                                   (voice, row_id, voice)).rowcount
            self._conn.commit()
        return n == 1

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
                   file_path: Optional[str] = None, billed_chars: Optional[int] = None,
                   voice: Optional[str] = None, model: Optional[str] = None,
                   style: Optional[str] = None, check=None,
                   qc_takes: Optional[int] = None, clip_id: Optional[str] = None) -> None:
        # A status-only update keeps the voice. `model` (a generation result)
        # writes model and style together, style as given (None: no direction);
        # without it both are kept. `check` (qc.Check), `qc_takes` and
        # `clip_id` (phrases) likewise.
        more_sql, more_params = "", []
        if check is not None:
            more_sql += "review = ?, qc_seconds = ?, qc_end_db = ?, "
            more_params += [check.review, check.seconds, check.end_db]
        if qc_takes is not None:
            more_sql += "qc_takes = ?, "
            more_params.append(qc_takes)
        if clip_id is not None:
            more_sql += "clip_id = ?, "
            more_params.append(clip_id)
        with self._lock:
            self._conn.execute(
                f"UPDATE {self.table} SET status = ?, error = ?, file_path = ?, "
                "billed_chars = ?, voice = COALESCE(?, voice), "
                f"{_PROVENANCE_SQL}, {more_sql}updated_at = ? WHERE id = ?",
                (status, error, file_path, billed_chars, voice, model, model, style, *more_params,
                 time.time(), row_id),
            )
            self._conn.commit()

    def record_qc(self, results) -> None:
        """Store QC results [(row_id, qc.Check, qc_takes or None), ...] in one
        transaction (`wordbank qc` checks a whole bank); None keeps qc_takes."""
        now = time.time()
        with self._lock:
            self._conn.executemany(
                f"UPDATE {self.table} SET review = ?, qc_seconds = ?, qc_end_db = ?, "
                "qc_takes = COALESCE(?, qc_takes), updated_at = ? WHERE id = ?",
                [(c.review, c.seconds, c.end_db, takes, now, rid) for rid, c, takes in results],
            )
            self._conn.commit()

    def mark_submitted(self, row_ids, batch_id: str, *, voice: Optional[str] = None,
                       model: Optional[str] = None, style: Optional[str] = None) -> None:
        """Mark entries as enqueued in an async batch job (counts as one attempt;
        a QC re-take doesn't: its row was generated, it counts in qc_takes).
        `voice` (claiming an unattributed row) and `model`/`style` record the
        request now, since a later collect (possibly another process) doesn't
        know it. A row another voice's batch claimed meanwhile is left to it."""
        now = time.time()
        with self._lock:
            self._conn.executemany(
                f"UPDATE {self.table} SET status = ?, batch_id = ?, voice = COALESCE(voice, ?), "
                f"attempts = attempts + (status != '{RETAKE}'), {_PROVENANCE_SQL}, "
                "updated_at = ? WHERE id = ? AND (voice IS NULL OR voice = COALESCE(?, voice))",
                [(SUBMITTED, batch_id, voice, model, model, style, now, wid, voice)
                 for wid in row_ids],
            )
            self._conn.commit()

    def reset(self, row_ids) -> None:
        """Back to a fresh pending row (attempts 0, no error, no batch job, no
        QC result), for a regenerate. file_path and provenance stay until the
        next take."""
        now = time.time()
        with self._lock:
            self._conn.executemany(
                f"UPDATE {self.table} SET status = ?, attempts = 0, error = NULL, "
                "batch_id = NULL, review = NULL, qc_seconds = NULL, qc_end_db = NULL, "
                "qc_takes = 0, qc_accepted = 0, updated_at = ? WHERE id = ?",
                [(PENDING, now, rid) for rid in row_ids],
            )
            self._conn.commit()

    def discard_unfinished(self, row_id: int) -> None:
        """Delete the row unless it holds a clip (a save that failed midway)."""
        with self._lock:
            self._conn.execute(f"DELETE FROM {self.table} WHERE id = ? AND status != ?",
                               (row_id, DONE))
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

    def qc_counts(self) -> dict:
        """{"flagged": clips with a QC flag (ready or queued for a re-take),
        "unchecked": ready clips never QC-checked (e.g. generated before QC),
        "accepted": clips a person kept despite a flag}. A RETAKE row always
        carries its take's measurements."""
        with self._lock:
            flagged, unchecked, accepted = self._conn.execute(
                f"SELECT COUNT(review), SUM(qc_end_db IS NULL AND qc_accepted = 0), SUM(qc_accepted) "
                f"FROM {self.table} WHERE status IN (?, ?)", (DONE, RETAKE)).fetchone()
        return {"flagged": flagged, "unchecked": unchecked or 0, "accepted": accepted or 0}

    def accept_qc(self, row_ids) -> None:
        """A person listened and kept these clips: clear their flag and mark them
        so `wordbank qc` neither re-flags nor re-takes them (reset() undoes it)."""
        now = time.time()
        with self._lock:
            self._conn.executemany(
                f"UPDATE {self.table} SET review = NULL, qc_accepted = 1, updated_at = ? WHERE id = ?",
                [(now, rid) for rid in row_ids])
            self._conn.commit()

    def iter_rows(self):
        with self._lock:
            return self._conn.execute(
                f"SELECT * FROM {self.table} ORDER BY id"
            ).fetchall()

    def entries(self, text: str, lang: str, voice: Optional[str] = None) -> list:
        """Every row of (text_col, lang), in id order; `voice` pins one voice.
        Not status-scoped, unlike BankLookup.get*."""
        sql = f"SELECT * FROM {self.table} WHERE {self.text_col} = ? AND lang = ?"
        params = [text, lang]
        if voice is not None:
            sql += " AND voice = ?"
            params.append(voice)
        with self._lock:
            return self._conn.execute(sql + " ORDER BY id", params).fetchall()

    def get_entry(self, text: str, lang: str, voice: Optional[str] = None):
        """The first of entries() (lowest id), or None. Used by writers to find
        a row they just inserted (pass the voice: a text may have several)."""
        rows = self.entries(text, lang, voice)
        return rows[0] if rows else None
