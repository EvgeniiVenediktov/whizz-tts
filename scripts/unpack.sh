#!/usr/bin/env bash
# Unpack + set up the bank server (serving-only deps — no ML stack).
#
#   bash unpack.sh ARCHIVE.tar.gz [DEST]   # extract the tarball, then set up
#   scripts/unpack.sh                      # run inside an already-extracted tree
#
# Creates .venv, installs requirements-base.txt, and verifies the bank. The
# archive keeps wordbank.db / output/words_ogg at the paths the server defaults
# to (BANK_DB, OGG_ROOT), so no env config is needed.
set -euo pipefail

archive="${1:-}"
dest="${2:-tts-bank}"
if [ -n "$archive" ] && [ -f "$archive" ]; then
  echo "extracting $archive → $dest/"
  mkdir -p "$dest"
  tar -xzf "$archive" -C "$dest"
  cd "$dest"
else
  cd "$(dirname "$0")/.."               # already extracted — go to the root
fi

command -v python3 >/dev/null || { echo "error: python3 not found" >&2; exit 1; }
echo "creating .venv + installing serving deps…"
python3 -m venv .venv
./.venv/bin/python -m pip install -q -U pip
./.venv/bin/python -m pip install -q -r requirements-base.txt

# Sanity: DB readable + OGG present (normal connection + query_only, never mode=ro).
./.venv/bin/python - <<'PY'
import sqlite3, pathlib
db = sqlite3.connect("wordbank.db")
db.execute("PRAGMA query_only = ON")
w = db.execute("SELECT COUNT(*) FROM words WHERE status IN ('done','verified')").fetchone()[0]
try:
    p = db.execute("SELECT COUNT(*) FROM phrases WHERE status IN ('done','verified')").fetchone()[0]
except sqlite3.OperationalError:
    p = 0
ogg = sum(1 for _ in pathlib.Path("output/words_ogg").rglob("*.ogg"))
print(f"  bank ready: {w} words, {p} phrases, {ogg} OGG files on disk")
db.close()
PY

echo
echo "run the server:"
echo "  SERVE_FRONTEND=1 .venv/bin/uvicorn src.server:app --port 8000"
echo "  → open http://localhost:8000"
echo "live synthesis (optional): MODEL_ENABLED=1 + .venv/bin/pip install -r requirements-model.txt"
