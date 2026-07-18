"""Smoke-test the Gemini Batch + TTS path on a few words BEFORE a bulk run.

Confirms, against a real key:
  - file-based batch accepts the per-request TTS generationConfig;
  - results carry base64 PCM keyed back to each request;
  - the voice instruction is NOT spoken (listen to the WAVs);
  - audio decodes and writes to WAV (no ffmpeg).

Requires GEMINI_API_KEY (or GOOGLE_API_KEY).

  python scripts/gemini_batch_smoke.py --voice Kore
"""
import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from wordbank.gemini_batch import (
    BATCH_DEFAULT_MODEL, VOICE_INSTRUCTION, GeminiBatch, audio_from_line, build_jsonl,
    write_wav_from_pcm_bytes,
)

WORDS = ["cat", "elephant", "rhythm", "umbrella", "two"]
OUT = pathlib.Path(__file__).parent.parent / "output" / "gemini_batch_smoke"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--voice", default="Kore")
    p.add_argument("--model", default=BATCH_DEFAULT_MODEL)
    p.add_argument("--poll-interval", type=int, default=15)
    args = p.parse_args()

    batch = GeminiBatch(model=args.model, voice=args.voice)
    rows = [{"id": i, "word": w} for i, w in enumerate(WORDS, start=1)]
    by_key = {str(r["id"]): r["word"] for r in rows}

    job_name = batch.submit_file(build_jsonl(rows, args.voice, VOICE_INSTRUCTION),
                                 display_name="wordbank-smoke")
    print(f"submitted job: {job_name}")

    while True:
        job = batch.poll(job_name)
        if batch.is_done(job):
            print(f"final state: {getattr(job, 'state', '?')}")
            break
        print(f"state: {getattr(job, 'state', '?')} — waiting {args.poll_interval}s")
        time.sleep(args.poll_interval)

    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for line in batch.result_lines(job):
        key = str(line.get("key"))
        word = by_key.get(key, key)
        if line.get("error"):
            print(f"  {word!r}: ERROR {line['error']}")
            continue
        pcm = audio_from_line(line)
        if not pcm:
            print(f"  {word!r}: NO AUDIO")
            continue
        write_wav_from_pcm_bytes(OUT / f"{word}.wav", pcm)
        n += 1
        print(f"  {word!r}: {len(pcm)} PCM bytes -> {word}.wav  OK")
    print(f"\nwrote {n}/{len(WORDS)} clips to {OUT}")


if __name__ == "__main__":
    main()
