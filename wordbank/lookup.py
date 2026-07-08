"""Read-only word→file lookup over the manifest. Stdlib sqlite3 only.

Exact lookup is an index hit on the manifest's UNIQUE(word, lang) B-tree —
O(log n), no extra service. Prefix autocomplete uses the same index via a range
scan (word >= prefix AND word < prefix++), so it stays index-backed too.
"""
import os
import sqlite3
import threading
from typing import List, Optional

_READY = ("done", "verified")
_READY_SQL = "status IN ({})".format(", ".join(f"'{s}'" for s in _READY))


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
            self._conn = conn
            # Capabilities (a bank from older code may lack these).
            self._words_has_voice = any(
                r[1] == "voice" for r in conn.execute("PRAGMA table_info(words)")
            )
            self._has_phrases = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='phrases'"
            ).fetchone() is not None

    @property
    def available(self) -> bool:
        with self._lock:
            self._open()
            return self._conn is not None

    @property
    def path(self) -> Optional[str]:
        return os.path.abspath(self._db_path) if self._db_path else None

    def count(self) -> int:
        """Number of words ready to serve (done/verified). Memoized — the served
        bank is read-only and static per process, so /health stays cheap."""
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
        """Languages the bank can serve: distinct `lang` codes with ready counts."""
        with self._lock:
            self._open()
            if self._conn is None:
                return []
            rows = self._conn.execute(
                f"SELECT lang, COUNT(*) FROM words WHERE {_READY_SQL} "
                "GROUP BY lang ORDER BY lang"
            ).fetchall()
        return [{"code": r[0], "words": r[1]} for r in rows]

    def get(self, word: str, lang: str = "en") -> Optional[sqlite3.Row]:
        """Exact word match (case-insensitive). Returns the row or None."""
        w = word.strip().lower()
        with self._lock:
            self._open()
            if self._conn is None:
                return None
            voice = "voice" if self._words_has_voice else "NULL AS voice"
            return self._conn.execute(
                f"SELECT word, lang, file_path, status, {voice} FROM words "
                f"WHERE word = ? AND lang = ? AND {_READY_SQL}",
                (w, lang),
            ).fetchone()

    def get_phrase(self, text: str, lang: str = "en",
                   voice: Optional[str] = None) -> Optional[sqlite3.Row]:
        """Exact phrase match (case-insensitive). If voice is omitted, returns any
        voice for that phrase. Returns the row or None (also None if no phrases)."""
        with self._lock:
            self._open()
            if self._conn is None or not self._has_phrases:
                return None
            sql = ("SELECT text, lang, file_path, status, voice, seed, "
                   "ref_audio, ref_text FROM phrases "
                   f"WHERE LOWER(text) = LOWER(?) AND lang = ? AND {_READY_SQL}")
            params = [text.strip(), lang]
            if voice is not None:
                sql += " AND voice = ?"
                params.append(voice)
            return self._conn.execute(sql + " LIMIT 1", params).fetchone()

    def search(self, prefix: str, lang: str = "en", limit: int = 10,
               kind: Optional[str] = None) -> List[dict]:
        """Merged prefix search over words + phrases. Each result is tagged with
        its `type` ('word'|'phrase') and `voice`. `kind` optionally restricts to one."""
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
                parts.append(
                    f"SELECT word AS text, 'word' AS type, {vw} AS voice FROM words "
                    f"WHERE lang = ? AND word >= ? AND word < ? AND {_READY_SQL}")
                params += [lang, p, _prefix_upper(p)]
            if kind in (None, "phrase") and self._has_phrases:
                parts.append(
                    "SELECT text, 'phrase' AS type, voice FROM phrases "
                    f"WHERE lang = ? AND LOWER(text) LIKE ? AND {_READY_SQL}")
                params += [lang, p + "%"]
            if not parts:
                return []
            sql = " UNION ALL ".join(parts) + " ORDER BY text LIMIT ?"
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

    def suggest(self, prefix: str, lang: str = "en", limit: int = 10) -> List[str]:
        """Index-backed prefix autocomplete; returns up to `limit` words."""
        p = prefix.strip().lower()
        if not p:
            return []
        with self._lock:
            self._open()
            if self._conn is None:
                return []
            rows = self._conn.execute(
                f"SELECT word FROM words WHERE lang = ? AND word >= ? AND word < ? "
                f"AND {_READY_SQL} ORDER BY word LIMIT ?",
                (lang, p, _prefix_upper(p), limit),
            ).fetchall()
        return [r[0] for r in rows]

    def voices(self, lang: Optional[str] = None) -> List[dict]:
        """Registered voice-clone references (optionally one language). Empty if
        the `voices` table doesn't exist yet (older bank)."""
        with self._lock:
            self._open()
            if self._conn is None:
                return []
            sql = ("SELECT name, lang, ref_audio, ref_text, instruct, source "
                   "FROM voices")
            params: tuple = ()
            if lang is not None:
                sql += " WHERE lang = ?"
                params = (lang,)
            sql += " ORDER BY lang, name"
            try:
                rows = self._conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                return []
        return [dict(r) for r in rows]

    def get_voice(self, name: str, lang: str) -> Optional[sqlite3.Row]:
        """One reference by (name, lang). None if absent or no `voices` table."""
        with self._lock:
            self._open()
            if self._conn is None:
                return None
            try:
                return self._conn.execute(
                    "SELECT name, lang, ref_audio, ref_text, instruct FROM voices "
                    "WHERE name = ? AND lang = ?", (name, lang),
                ).fetchone()
            except sqlite3.OperationalError:
                return None

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            self._ready_count = None
