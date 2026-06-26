"""One-time migration: make manifest file_path values relative to the bank root.

Old rows stored e.g. "output/words/en/gemini-Kore/1/1914_hello.wav" (relative to
the generation CWD). The server now resolves paths under WAV_ROOT/OGG_ROOT, so
file_path must be relative to the bank root: "en/gemini-Kore/1/1914_hello.wav".

Idempotent — rows already relative are left alone.

  python scripts/migrate_manifest_paths.py [wordbank.db] [output/words]
"""
import os
import sqlite3
import sys


def main() -> None:
    db = sys.argv[1] if len(sys.argv) > 1 else "wordbank.db"
    root = sys.argv[2] if len(sys.argv) > 2 else "output/words"
    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT id, file_path FROM words WHERE file_path IS NOT NULL"
    ).fetchall()
    changed = 0
    for wid, fp in rows:
        if os.path.isabs(fp):
            rel = os.path.relpath(fp, os.path.abspath(root))
        elif fp.startswith(root.rstrip("/") + os.sep):
            rel = os.path.relpath(fp, root)
        else:
            rel = fp  # already relative to the bank root
        if rel != fp and not rel.startswith(".."):
            conn.execute("UPDATE words SET file_path = ? WHERE id = ?", (rel, wid))
            changed += 1
    conn.commit()
    conn.close()
    print(f"migrated {changed} of {len(rows)} paths (root={root!r})")


if __name__ == "__main__":
    main()
