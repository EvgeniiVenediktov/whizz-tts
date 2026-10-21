"""Encode WAV masters into a parallel OGG/Opus copy tree (originals untouched).

  output/words/...           (WAV masters — left as-is)
  output/words_ogg/...       (OGG copies, mirrored structure)

Defaults: 16 kbps, libopus VoIP mode (tuned for speech) → ~3 KB/word. ffmpeg work
is parallelized; existing copies are skipped unless --force. Prints space saved.

  python scripts/encode_wavs.py output/words output/words_ogg              # 16k voip
  python scripts/encode_wavs.py output/words output/words_ogg --force      # re-encode all
  python scripts/encode_wavs.py output/words output/words_ogg --bitrate 32k --application audio
"""
import argparse
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from wordbank.constants import BANK_OGG_APPLICATION, BANK_OGG_BITRATE
from wordbank.paths import mirror_path


def encode_one(wav: pathlib.Path, src_root: str, dest_root: str, bitrate: str,
               application: str = BANK_OGG_APPLICATION, force: bool = False):
    dest = mirror_path(str(wav), src_root, dest_root, ".ogg")
    if dest is None:
        return (str(wav), 0, 0, "outside src root")
    dest = pathlib.Path(dest)
    wav_bytes = wav.stat().st_size
    if dest.exists() and not force:
        return (str(wav), wav_bytes, dest.stat().st_size, "skip")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav), "-c:a", "libopus",
             "-application", application, "-b:a", bitrate, "-f", "ogg", str(dest)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
    except subprocess.CalledProcessError as e:
        return (str(wav), wav_bytes, 0, f"ffmpeg failed ({e.returncode})")
    return (str(wav), wav_bytes, dest.stat().st_size, "ok")


def main() -> None:
    p = argparse.ArgumentParser(description="Mirror WAV masters into an OGG/Opus copy tree")
    p.add_argument("src", help="root of WAV masters (left untouched)")
    p.add_argument("dest", help="root for the OGG copies (mirrored structure)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--bitrate", default=BANK_OGG_BITRATE)
    p.add_argument("--application", default=BANK_OGG_APPLICATION,
                   choices=["voip", "audio", "lowdelay"],
                   help="libopus mode; 'voip' is tuned for speech at low bitrates")
    p.add_argument("--force", action="store_true",
                   help="re-encode even if the OGG copy already exists")
    args = p.parse_args()

    wavs = sorted(pathlib.Path(args.src).rglob("*.wav"))
    if not wavs:
        print(f"no .wav files under {args.src}")
        return
    print(f"encoding {len(wavs)} WAV(s) -> {args.dest} "
          f"({args.bitrate} {args.application}, {args.workers} workers)…")

    ok = failed = 0
    wav_total = ogg_total = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for path, wb, ob, note in pool.map(
            lambda w: encode_one(w, args.src, args.dest, args.bitrate,
                                 args.application, args.force), wavs
        ):
            if note in ("ok", "skip"):
                ok += 1
                wav_total += wb
                ogg_total += ob
            else:
                failed += 1
                print(f"  FAIL {path}: {note}")

    mb = 1024 * 1024
    ratio = (wav_total / ogg_total) if ogg_total else 0
    print(f"done: {ok} ok, {failed} failed")
    print(f"WAV {wav_total/mb:.1f} MB  ->  OGG {ogg_total/mb:.1f} MB  ({ratio:.1f}x smaller)")


if __name__ == "__main__":
    main()
