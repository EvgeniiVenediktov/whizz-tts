"""Synchronous Gemini TTS — one synthesize_pcm call per word, write WAV.

A billing probe outside the manifest: if this also fails with
FAILED_PRECONDITION, the key's project needs billing/paid tier enabled (Batch
API + TTS audio output require it; free tier has region limits). To fill or
repair the bank synchronously, use `python -m wordbank gemini-run` instead.

VOICE_INSTRUCTION is prepended only for models that keep it unspoken (3.1/2.5);
the 3.8 models get the bare word, or --style as speech metadata.

  GEMINI_API_KEY=… python scripts/gemini_sync.py --words words10.txt --voice Kore
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from wordbank.gemini_batch import VOICE_INSTRUCTION, write_wav_from_pcm_bytes
from wordbank.gemini_tts import DEFAULT_MODEL as DEFAULT_SYNC_MODEL, model_caps, synthesize_pcm
from wordbank.paths import safe_filename


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--words", required=True, help="text file, one word per line")
    p.add_argument("--voice", default="Kore")
    p.add_argument("--model", default=DEFAULT_SYNC_MODEL)
    p.add_argument("--style", default=None, help="speech-metadata style (3.8 models only)")
    p.add_argument("--out", default="output/words_sync")
    args = p.parse_args()
    instruct = VOICE_INSTRUCTION if not model_caps(args.model).style else ""

    words = [w.strip() for w in open(args.words, encoding="utf-8")
             if w.strip() and not w.startswith("#")]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ok = 0
    for i, word in enumerate(words, start=1):
        try:
            data = synthesize_pcm(word, voice=args.voice, instruct=instruct,
                                  model=args.model, style=args.style)
        except Exception as e:
            print(f"  {word!r}: ERROR {type(e).__name__}: {str(e)[:160]}")
            continue
        path = out / f"{i}_{safe_filename(word)}.wav"
        write_wav_from_pcm_bytes(path, data)
        ok += 1
        print(f"  {word!r}: {len(data)} PCM bytes -> {path}")
    print(f"\nwrote {ok}/{len(words)} clips to {out}")


if __name__ == "__main__":
    main()
