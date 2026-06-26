"""Synchronous Gemini TTS — one generate_content call per word, write WAV.

Use for small sets (no batch needed) and as a billing probe: if this also fails
with FAILED_PRECONDITION, the key's project needs billing/paid tier enabled
(Batch API + TTS audio output require it; free tier has region limits).

  GEMINI_API_KEY=… python scripts/gemini_sync.py --words words10.txt --voice Kore
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from google import genai
from google.genai import types

from wordbank.gemini_batch import VOICE_INSTRUCTION, write_wav
from wordbank.pipeline import _safe

# generateContent-capable TTS model (not all support batch; this one is fine for sync)
DEFAULT_SYNC_MODEL = "gemini-2.5-flash-preview-tts"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--words", required=True, help="text file, one word per line")
    p.add_argument("--voice", default="Kore")
    p.add_argument("--model", default=DEFAULT_SYNC_MODEL)
    p.add_argument("--out", default="output/words_sync")
    args = p.parse_args()

    words = [w.strip() for w in open(args.words, encoding="utf-8")
             if w.strip() and not w.startswith("#")]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    client = genai.Client()
    cfg = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=args.voice))),
    )

    ok = 0
    for i, word in enumerate(words, start=1):
        try:
            resp = client.models.generate_content(
                model=args.model,
                contents=f"{VOICE_INSTRUCTION} {word}",
                config=cfg,
            )
            data = resp.candidates[0].content.parts[0].inline_data.data
        except Exception as e:
            print(f"  {word!r}: ERROR {type(e).__name__}: {str(e)[:160]}")
            continue
        path = out / f"{i}_{_safe(word)}.wav"
        write_wav(path, data)
        ok += 1
        print(f"  {word!r}: {len(data)} PCM bytes -> {path}")
    print(f"\nwrote {ok}/{len(words)} clips to {out}")


if __name__ == "__main__":
    main()
