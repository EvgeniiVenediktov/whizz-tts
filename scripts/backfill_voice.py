"""Backfill the `voice` column on existing words/phrases rows.

Voice was previously encoded only in the file_path directory segment
(`<lang>/<backend>-<voice>/<shard>/<id>_<text>.wav`). This parses it back out
and fills any NULL `voice`. Idempotent — only touches NULLs, never overwrites.

  python scripts/backfill_voice.py [--db wordbank.db]
"""
import argparse
import pathlib
import sqlite3
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from wordbank.paths import voice_from_path  # noqa: E402


def backfill(conn: sqlite3.Connection, table: str) -> tuple[int, int]:
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN voice TEXT")
    except sqlite3.OperationalError:
        pass  # column exists
    rows = conn.execute(
        f"SELECT id, file_path FROM {table} "
        "WHERE voice IS NULL AND file_path IS NOT NULL"
    ).fetchall()
    filled = 0
    for rid, fp in rows:
        v = voice_from_path(fp)
        if v:
            conn.execute(f"UPDATE {table} SET voice = ? WHERE id = ?", (v, rid))
            filled += 1
    conn.commit()
    return filled, len(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="wordbank.db")
    args = p.parse_args()
    if not pathlib.Path(args.db).exists():
        print(f"no DB at {args.db} — nothing to do")
        return
    conn = sqlite3.connect(args.db)
    for table in ("words", "phrases"):
        try:
            filled, candidates = backfill(conn, table)
            print(f"{table}: filled {filled}/{candidates} NULL-voice rows")
        except sqlite3.OperationalError as e:
            print(f"{table}: skipped ({e})")
    conn.close()


if __name__ == "__main__":
    main()
