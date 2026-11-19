"""Read-only word→file lookup over the manifest. Stdlib sqlite3 only.

Exact lookup is an index hit on the manifest's UNIQUE(lang, word, voice) B-tree
(UNIQUE(word, lang) + idx_words_lang_word on banks the key migration hasn't
touched) — O(log n), no extra service. Prefix autocomplete uses the same index
via a range scan (word >= prefix AND word < prefix++), so it stays index-backed
too.

A word may be banked in several voices (one row each). With a voice, lookups
match that voice only; without one, get() serves the caller's preferred voice
if banked, else the earliest banked clip (lowest id), and searches list one
result per voice.

Phrases carry the same voice labels as words ("Kore"). Phrases saved before
that say "gemini-Kore" until a Manifest opens the bank and relabels them
(Manifest._migrate_phrase_voices); this reader never migrates, so phrase
lookups read either spelling as the label the migration gives it.
"""
import os
import sqlite3
import threading
from typing import List, Optional

from .gemini_tts import plain_voice_label
from .paths import voice_from_path

_READY = ("done", "verified")
_READY_SQL = "status IN ({})".format(", ".join(f"'{s}'" for s in _READY))
# A phrase's voice as its label (see above); legacy rows only call into Python.
_PHRASE_VOICE_SQL = ("CASE WHEN voice GLOB 'gemini-?*' THEN plain_voice_label(voice) "
                     "ELSE voice END")


def _prefix_upper(prefix: str) -> str:
    """Smallest string strictly greater than every string starting with prefix."""
    return prefix[:-1] + chr(ord(prefix[-1]) + 1)


class BankLookup:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = None
        self._ready_count = None   # memoized; the served bank is static per process
        self._words_has_voice = False  # set on open (older banks may lack it)
        self._has_phrases = False
        self._open()

    def _open(self) -> None:
        """Open lazily (so a DB created after startup is picked up).

        NOTE: a normal connection — NOT mode=ro. A read-only connection on a WAL
        database uses a private in-memory -shm and therefore cannot see rows
        committed by another process (the batch writer) after it opened. A normal
        connection shares the real -shm and sees every commit; PRAGMA query_only
        keeps it read-only so we never write.
        """
        if self._conn is None and self._db_path and os.path.exists(self._db_path):
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only = ON")
            conn.create_function("plain_voice_label", 1, plain_voice_label, deterministic=True)
            self._conn = conn
            # Capabilities (a bank from older code may lack these).
            self._words_has_voice = any(
                r[1] == "voice" for r in conn.execute("PRAGMA table_info(words)")
            )
            self._has_phrases = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='phrases'"
            ).fetchone() is not None
            # Columns added to `phrases` over time (voice, seed, ref_audio,
            # ref_text). Manifest ALTER-migrates on open, but this reader must
            # serve old banks it never writes — select NULL for missing ones.
            self._phrase_cols = {
                r[1] for r in conn.execute("PRAGMA table_info(phrases)")
            } if self._has_phrases else set()

    @property
    def available(self) -> bool:
        with self._lock:
            self._open()
            return self._conn is not None

    @property
    def path(self) -> Optional[str]:
        return os.path.abspath(self._db_path) if self._db_path else None

    def count(self) -> int:
        """Number of word clips ready to serve (done/verified; a word in two
        voices counts twice). Memoized — the served bank is read-only and
        static per process, so /health stays cheap."""
        with self._lock:
            if self._ready_count is not None:
                return self._ready_count
            self._open()
            if self._conn is None:
                return 0
            self._ready_count = self._conn.execute(
                f"SELECT COUNT(*) FROM words WHERE {_READY_SQL}"
            ).fetchone()[0]
            return self._ready_count

    def languages(self) -> List[dict]:
        """Languages the bank can serve: distinct `lang` codes with ready counts
        (clips: a word in two voices counts twice)."""
        with self._lock:
            self._open()
            if self._conn is None:
                return []
            rows = self._conn.execute(
                f"SELECT lang, COUNT(*) FROM words WHERE {_READY_SQL} "
                "GROUP BY lang ORDER BY lang"
            ).fetchall()
        return [{"code": r[0], "words": r[1]} for r in rows]

    def _word_voice_sql(self, voice: str):
        """WHERE fragment + params for words banked in `voice`. Rows without a
        recorded voice (banks predating the column, or not yet migrated) match
        on the voice in their file_path (`<lang>/gemini-<voice>/…`)."""
        segment = f"gemini-{voice}/"
        by_path = "substr(file_path, instr(file_path, '/') + 1, ?) = ?"
        if not self._words_has_voice:
            return by_path, [len(segment), segment]
        return f"(voice = ? OR (voice IS NULL AND {by_path}))", [voice, len(segment), segment]

    def get(self, word: str, lang: str = "en", voice: Optional[str] = None,
            prefer: Optional[str] = None) -> Optional[sqlite3.Row]:
        """Exact word match (case-insensitive). Returns the row or None.

        If `voice` (a bank label, e.g. "Kore", "en-gb-tutor-11") is given, the
        word must have been generated with it. Rows without a recorded voice
        (banks predating the column, or not yet backfilled) fall back to the
        voice in their file_path. Without `voice`, the clip in `prefer` (the
        server's default voice) if there is one, else the earliest banked
        (lowest id), so the choice is stable.
        """
        w = word.strip().lower()
        with self._lock:
            self._open()
            if self._conn is None:
                return None
            cols = "voice" if self._words_has_voice else "NULL AS voice"
            rows = self._conn.execute(
                f"SELECT word, lang, file_path, status, {cols} FROM words "
                f"WHERE word = ? AND lang = ? AND {_READY_SQL} ORDER BY id",
                (w, lang),
            ).fetchall()

        def voiced(label):
            return next((r for r in rows
                         if (r["voice"] or voice_from_path(r["file_path"] or "")) == label), None)
        if voice is not None:
            return voiced(voice)
        return (voiced(prefer) if prefer else None) or (rows[0] if rows else None)

    def get_phrase(self, text: str, lang: str = "en",
                   voice: Optional[str] = None) -> Optional[sqlite3.Row]:
        """Exact phrase match (case-insensitive). `voice` (a bank label, "Kore")
        pins it, matching a legacy "gemini-Kore" row too; omitted, any voice
        (the earliest banked, so the choice is stable). Returns the row, its
        voice as the label, or None (also None if no phrases)."""
        with self._lock:
            self._open()
            if self._conn is None or not self._has_phrases:
                return None
            if voice is not None and "voice" not in self._phrase_cols:
                return None  # voice-pinned lookup can't match a voiceless bank
            cols = ", ".join(
                f"NULL AS {c}" if c not in self._phrase_cols
                else f"{_PHRASE_VOICE_SQL} AS voice" if c == "voice" else c
                for c in ("voice", "seed", "ref_audio", "ref_text")
            )
            sql = (f"SELECT text, lang, file_path, status, {cols} FROM phrases "
                   f"WHERE LOWER(text) = LOWER(?) AND lang = ? AND {_READY_SQL}")
            params = [text.strip(), lang]
            if voice is not None:
                sql += f" AND {_PHRASE_VOICE_SQL} = ?"
                params.append(voice)
            return self._conn.execute(sql + " ORDER BY id LIMIT 1", params).fetchone()

    def search(self, prefix: str, lang: str = "en", limit: int = 10,
               kind: Optional[str] = None, voice: Optional[str] = None) -> List[dict]:
        """Merged prefix search over words + phrases. Each result is tagged with
        its `type` ('word'|'phrase') and `voice`, one per voice a text is banked
        in. `kind` optionally restricts to one; `voice` (a bank label, "Kore")
        to words and phrases in that voice, as /v1/speak matches them."""
        p = prefix.strip().lower()
        if not p:
            return []
        with self._lock:
            self._open()
            if self._conn is None:
                return []
            parts, params = [], []
            if kind in (None, "word"):
                vw = "voice" if self._words_has_voice else "NULL"
                sql = (f"SELECT word AS text, 'word' AS type, {vw} AS voice FROM words "
                       f"WHERE lang = ? AND word >= ? AND word < ? AND {_READY_SQL}")
                params += [lang, p, _prefix_upper(p)]
                if voice is not None:
                    where, more = self._word_voice_sql(voice)
                    sql += f" AND {where}"
                    params += more
                parts.append(sql)
            if kind in (None, "phrase") and self._has_phrases and (
                    voice is None or "voice" in self._phrase_cols):
                vp = _PHRASE_VOICE_SQL if "voice" in self._phrase_cols else "NULL"
                # DISTINCT: a duplicate the migration kept under a legacy spelling
                # is the same text in the same voice
                sql = (f"SELECT DISTINCT text, 'phrase' AS type, {vp} AS voice FROM phrases "
                       f"WHERE lang = ? AND LOWER(text) LIKE ? AND {_READY_SQL}")
                params += [lang, p + "%"]
                if voice is not None:
                    sql += f" AND {_PHRASE_VOICE_SQL} = ?"
                    params.append(voice)
                parts.append(sql)
            if not parts:
                return []
            sql = " UNION ALL ".join(parts) + " ORDER BY text, voice LIMIT ?"
            params.append(limit)
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def phrase_languages(self) -> List[dict]:
        """Languages the phrase bank can serve: distinct lang codes with ready counts."""
        with self._lock:
            self._open()
            if self._conn is None or not self._has_phrases:
                return []
            rows = self._conn.execute(
                f"SELECT lang, COUNT(*) FROM phrases WHERE {_READY_SQL} "
                "GROUP BY lang ORDER BY lang"
            ).fetchall()
        return [{"code": r[0], "words": r[1]} for r in rows]

    def suggest(self, prefix: str, lang: str = "en", limit: int = 10,
                voice: Optional[str] = None) -> List[str]:
        """Index-backed prefix autocomplete; returns up to `limit` distinct
        words (any voice, or banked in `voice`)."""
        p = prefix.strip().lower()
        if not p:
            return []
        with self._lock:
            self._open()
            if self._conn is None:
                return []
            sql = (f"SELECT DISTINCT word FROM words WHERE lang = ? AND word >= ? AND word < ? "
                   f"AND {_READY_SQL}")
            params = [lang, p, _prefix_upper(p)]
            if voice is not None:
                where, more = self._word_voice_sql(voice)
                sql += f" AND {where}"
                params += more
            rows = self._conn.execute(sql + " ORDER BY word LIMIT ?", params + [limit]).fetchall()
        return [r[0] for r in rows]

    def word_voices(self) -> List[dict]:
        """Ready word clips per language and voice (NULL: not recorded)."""
        with self._lock:
            self._open()
            if self._conn is None:
                return []
            vw = "voice" if self._words_has_voice else "NULL"
            rows = self._conn.execute(
                f"SELECT lang, {vw} AS v, COUNT(*) FROM words WHERE {_READY_SQL} "
                "GROUP BY lang, v ORDER BY lang, v"
            ).fetchall()
        return [{"code": r[0], "voice": r[1], "words": r[2]} for r in rows]

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            self._ready_count = None
