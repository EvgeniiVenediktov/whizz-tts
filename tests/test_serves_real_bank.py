"""End-to-end serving check against the REAL local bank.

The other server tests use synthetic in-process data, so they miss regressions
that only show against the real wordbank.db + OGG tree (path resolution, WAL
read visibility, env-default drift). This boots the actual app over the real
bank and asserts a known word truly serves OGG audio.

Runs in the normal suite; SKIPS (not fails) when no local bank is present.
Run it after any change to src/server.py or wordbank/lookup.py.
"""
import os
import pathlib
import sqlite3

import pytest
from fastapi.testclient import TestClient

BANK_DB = os.getenv("BANK_DB", "wordbank.db")
OGG_ROOT = os.getenv("OGG_ROOT", "output/words_ogg")

pytestmark = pytest.mark.skipif(
    not (pathlib.Path(BANK_DB).exists() and pathlib.Path(OGG_ROOT).is_dir()),
    reason="no local bank (wordbank.db + output/words_ogg) — skipping real-bank serving check",
)


def _pick_servable_word():
    """A done word whose OGG copy is actually on disk (tolerates partial encodes)."""
    conn = sqlite3.connect(BANK_DB)
    try:
        rows = conn.execute(
            "SELECT word, lang, file_path FROM words "
            "WHERE status IN ('done','verified') LIMIT 500"
        ).fetchall()
    finally:
        conn.close()
    for word, lang, fp in rows:
        ogg = pathlib.Path(OGG_ROOT) / (os.path.splitext(fp)[0] + ".ogg")
        if ogg.exists():
            return word, lang
    return None


def test_real_bank_actually_serves_audio():
    sample = _pick_servable_word()
    if sample is None:
        pytest.skip("no encoded OGG found in the local bank")
    word, lang = sample

    from src.server import app
    with TestClient(app) as c:
        health = c.get("/health").json()
        assert health["bank"] is True
        assert health["bank_words"] > 0, "manifest reports zero ready words"

        # The actual thing that kept breaking: the clip must come back as real audio.
        r = c.get(f"/v1/bank/{word}", params={"lang": lang})
        assert r.status_code == 200, f"{word!r} ({lang}) not served: {r.status_code} {r.text}"
        assert r.headers["content-type"].startswith("audio/ogg")
        assert r.content[:4] == b"OggS", "response is not OGG audio"
        assert len(r.content) > 500, f"suspiciously tiny clip ({len(r.content)} B)"

        # Lookup + languages must surface it too.
        sugg = c.get("/v1/bank", params={"prefix": word[:3], "lang": lang}).json()["words"]
        assert word in sugg, f"{word!r} not returned by prefix search"
        langs = c.get("/v1/languages").json()
        assert any(l["words"] > 0 for l in langs["wordbank"]), "no languages reported"
