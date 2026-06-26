"""Prepare the bank for upload — encode + stage into ONE folder. Does NOT upload.

Produces a single upload-ready directory (default dist/bank/) containing:
    words_ogg/      the served OGG tree (16k VoIP)
    wordbank.db     the checkpointed manifest

Upload its contents anywhere you like (S3 / GCS / rclone / rsync) yourself. The
deploy serves words_ogg/ + wordbank.db from wherever you put them.

  python scripts/package_bank.py
  python scripts/package_bank.py --out-dir /tmp/bank --bitrate 32k --application audio --force
"""
import argparse
import os
import pathlib
import shutil
import sqlite3
import subprocess
import sys

ROOT = pathlib.Path(__file__).parent.parent


def _stage_ogg(ogg_root: str, dst: pathlib.Path) -> int:
    """Mirror the OGG tree into dst via hardlinks (no extra disk); copy fallback."""
    root = pathlib.Path(ogg_root)
    n = 0
    for src in root.rglob("*.ogg"):
        out = dst / src.relative_to(root)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, out)
        except OSError:
            shutil.copy2(src, out)
        n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="Encode + stage the bank for manual upload (no upload performed)")
    p.add_argument("--wav-root", default="output/words", help="WAV masters (input)")
    p.add_argument("--ogg-root", default="output/words_ogg", help="OGG copies (encoded here)")
    p.add_argument("--db", default="wordbank.db", help="manifest")
    p.add_argument("--out-dir", default="dist/bank", help="staging dir you upload from")
    p.add_argument("--bitrate", default="16k")
    p.add_argument("--application", default="voip", choices=["voip", "audio", "lowdelay"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force", action="store_true", help="re-encode existing OGG copies")
    args = p.parse_args()

    # 1. Encode WAV -> OGG (served format).
    print(f"[1/3] encoding {args.wav_root} -> {args.ogg_root} ({args.bitrate} {args.application})")
    enc = [sys.executable, str(ROOT / "scripts/encode_wavs.py"),
           args.wav_root, args.ogg_root,
           "--bitrate", args.bitrate, "--application", args.application,
           "--workers", str(args.workers)]
    if args.force:
        enc.append("--force")
    if subprocess.run(enc).returncode != 0:
        sys.exit("encode step failed")

    # 2. Checkpoint the manifest so the staged .db is a single complete file.
    print(f"[2/3] checkpointing {args.db}")
    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    ready = conn.execute(
        "SELECT COUNT(*) FROM words WHERE status IN ('done','verified')"
    ).fetchone()[0]
    conn.close()

    # 3. Stage into one folder (OGG hardlinked → no disk bloat; manifest copied).
    out = pathlib.Path(args.out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    n = _stage_ogg(args.ogg_root, out / "words_ogg")
    shutil.copy2(args.db, out / "wordbank.db")
    print(f"[3/3] staged {n} OGG files + manifest -> {out}/")
    if n < ready:
        print(f"      WARNING: {n} OGG < {ready} ready words — bank incomplete; do not upload")

    print(f"\nReady. Upload the CONTENTS of {out}/ to any storage, e.g.:")
    print(f"  aws s3 sync {out}/  s3://BUCKET/PREFIX/")
    print(f"  rclone copy {out}/  REMOTE:PREFIX/")
    print(f"  gsutil -m rsync -r {out}/  gs://BUCKET/PREFIX/")
    print("Then point the deploy at PREFIX (it expects words_ogg/ + wordbank.db).")


if __name__ == "__main__":
    main()
