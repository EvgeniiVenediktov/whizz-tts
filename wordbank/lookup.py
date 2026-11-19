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

Phrases carry the same voice labels as words ("Kore"); older code saved them
as "gemini-Kore", which phrase lookups read as "Kore".
"""
import os
import sqlite3
import threading
from typing import List, Optional

from .gemini_tts import voice_label
from .paths import voice_from_path

_READY_SQL = "status = 'done'"
# A phrase's voice as its label (see above); legacy rows only call into Python.
_PHRASE_VOICE_SQL = ("CASE WHEN voice GLOB 'gemini-?*' THEN voice_label(voice) "
                     "ELSE voice END")


def _prefix_upper(prefix: str) -> str:
    """Smallest string strictly greater than every string starting with prefix.
    U+10FFFF (a noncharacter, in no word) can't be incremented: carry into the
    character before it; a prefix of nothing else gets "", which matches none.
    The surrogates (unencodable) are skipped."""
    head = prefix.rstrip("\U0010ffff")
    if not head:
        return ""
    nxt = ord(head[-1]) + 1
    return head[:-1] + chr(0xE000 if 0xD800 <= nxt < 0xE000 else nxt)


def _like_prefix(prefix: str) -> str:
    """A LIKE pattern (ESCAPE '\\') matching strings that start with `prefix`."""
    return prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


class BankLookup:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = None
        self._ready_count = None   # memoized; the served bank is static per process
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
            conn.create_function("voice_label", 1, voice_label, deterministic=True)
            self._conn = conn

    @property
    def available(self) -> bool:
        with self._lock:
            self._open()
            return self._conn is not None

    def _query(self, sql: str, params=()) -> list:
        """Rows of `sql`, or [] while there is no bank file."""
        with self._lock:
            self._open()
            return [] if self._conn is None else self._conn.execute(sql, params).fetchall()

    def count(self) -> int:
        """Word clips ready to serve (a word in two voices counts twice).
        Memoized: the served bank is static per process, so /health is cheap."""
        if self._ready_count is None:
            rows = self._query(f"SELECT COUNT(*) FROM words WHERE {_READY_SQL}")
            if not rows:
                return 0
            self._ready_count = rows[0][0]
        return self._ready_count

    def _lang_counts(self, table: str) -> List[dict]:
        return [{"code": lang, "words": n} for lang, n in self._query(
            f"SELECT lang, COUNT(*) FROM {table} WHERE {_READY_SQL} GROUP BY lang ORDER BY lang")]

    def languages(self) -> List[dict]:
        """Languages of the word bank, with ready clip counts."""
        return self._lang_counts("words")

    def phrase_languages(self) -> List[dict]:
        """Languages of the phrase bank, with ready counts."""
        return self._lang_counts("phrases")

    def word_voices(self) -> List[dict]:
        """Ready word clips per language and voice (NULL: not recorded)."""
        return [{"code": lang, "voice": v, "words": n} for lang, v, n in self._query(
            f"SELECT lang, voice, COUNT(*) FROM words WHERE {_READY_SQL} "
            "GROUP BY lang, voice ORDER BY lang, voice")]

    def _word_voice_sql(self, voice: str):
        """WHERE fragment + params for words banked in `voice`; a row without a
        recorded voice matches on its file_path (`<lang>/gemini-<voice>/…`)."""
        segment = f"gemini-{voice}/"
        return ("(voice = ? OR (voice IS NULL AND "
                "substr(file_path, instr(file_path, '/') + 1, ?) = ?))", [voice, len(segment), segment])

    def get(self, word: str, lang: str = "en", voice: Optional[str] = None,
            prefer: Optional[str] = None) -> Optional[sqlite3.Row]:
        """Exact word match (case-insensitive), or None. `voice` (a bank label,
        "Kore") pins it; else the `prefer` voice's clip, else the earliest
        banked (lowest id). A row without a recorded voice matches on its path."""
        rows = self._query(f"SELECT word, lang, file_path, status, voice FROM words "
                           f"WHERE word = ? AND lang = ? AND {_READY_SQL} ORDER BY id",
                           (word.strip().lower(), lang))

        def voiced(label):
            return next((r for r in rows
                         if (r["voice"] or voice_from_path(r["file_path"] or "")) == label), None)
        if voice is not None:
            return voiced(voice)
        return (voiced(prefer) if prefer else None) or (rows[0] if rows else None)

    def get_phrase(self, text: str, lang: str = "en",
                   voice: Optional[str] = None) -> Optional[sqlite3.Row]:
        """Exact phrase match (case-insensitive), the earliest banked, or None;
        `voice` pins it (a legacy "gemini-Kore" row matches "Kore")."""
        sql = (f"SELECT text, lang, file_path, status, {_PHRASE_VOICE_SQL} AS voice FROM phrases "
               f"WHERE LOWER(text) = LOWER(?) AND lang = ? AND {_READY_SQL}")
        params = [text.strip(), lang]
        if voice is not None:
            sql += f" AND {_PHRASE_VOICE_SQL} = ?"
            params.append(voice)
        rows = self._query(sql + " ORDER BY id LIMIT 1", params)
        return rows[0] if rows else None

    def search(self, prefix: str, lang: str = "en", limit: int = 10,
               kind: Optional[str] = None, voice: Optional[str] = None) -> List[dict]:
        """Merged prefix search over words + phrases, one result per voice a
        text is banked in, tagged `type` ('word'|'phrase') and `voice`. `kind`
        restricts to one; `voice` to that voice, as /v1/speak matches them."""
        p = prefix.strip().lower()
        parts, params = [], []
        if p and kind in (None, "word"):
            sql = (f"SELECT word AS text, 'word' AS type, voice FROM words "
                   f"WHERE lang = ? AND word >= ? AND word < ? AND {_READY_SQL}")
            params += [lang, p, _prefix_upper(p)]
            if voice is not None:
                where, more = self._word_voice_sql(voice)
                sql += f" AND {where}"
                params += more
            parts.append(sql)
        if p and kind in (None, "phrase"):
            # DISTINCT: one result for a text banked under both label spellings
            sql = (f"SELECT DISTINCT text, 'phrase' AS type, {_PHRASE_VOICE_SQL} AS voice FROM phrases "
                   f"WHERE lang = ? AND LOWER(text) LIKE ? ESCAPE '\\' AND {_READY_SQL}")
            params += [lang, _like_prefix(p)]
            if voice is not None:
                sql += f" AND {_PHRASE_VOICE_SQL} = ?"
                params.append(voice)
            parts.append(sql)
        if not parts:
            return []
        rows = self._query(" UNION ALL ".join(parts) + " ORDER BY text, voice LIMIT ?", params + [limit])
        return [dict(r) for r in rows]

    def suggest(self, prefix: str, lang: str = "en", limit: int = 10,
                voice: Optional[str] = None) -> List[str]:
        """Index-backed prefix autocomplete: up to `limit` distinct words (any
        voice, or banked in `voice`)."""
        p = prefix.strip().lower()
        if not p:
            return []
        sql = f"SELECT DISTINCT word FROM words WHERE lang = ? AND word >= ? AND word < ? AND {_READY_SQL}"
        params = [lang, p, _prefix_upper(p)]
        if voice is not None:
            where, more = self._word_voice_sql(voice)
            sql += f" AND {where}"
            params += more
        return [r[0] for r in self._query(sql + " ORDER BY word LIMIT ?", params + [limit])]

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            self._ready_count = None
