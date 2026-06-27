"""Generate an accent reference clip with Gemini — the local model's voice-clone
reference. OmniVoice then clones this accent/voice for the whole phrase bank.

Writes <out>/ref.wav + <out>/ref.txt (the spoken phrase, for --ref-text-file).
Needs GEMINI_API_KEY.

  GEMINI_API_KEY=… python scripts/make_accent_reference.py \
    --text "The quick brown fox jumps over the lazy dog." \
    --instruct "Speak in a British (Received Pronunciation) accent" \
    --voice Kore --out refs/en-gb
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from wordbank.gemini_batch import write_wav  # raw PCM → WAV


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True, help="reference phrase to speak")
    p.add_argument("--instruct", default="",
                   help="accent/style direction, e.g. 'Speak in a British accent' (not spoken)")
    p.add_argument("--voice", default="Kore", help="Gemini prebuilt voice")
    p.add_argument("--model", default="gemini-2.5-flash-preview-tts")
    p.add_argument("--out", default="refs/ref", help="output dir for ref.wav + ref.txt")
    args = p.parse_args()

    from google import genai
    from google.genai import types

    client = genai.Client()
    prompt = f"{args.instruct}: {args.text}" if args.instruct else args.text
    resp = client.models.generate_content(
        model=args.model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=args.voice))),
        ),
    )
    pcm = resp.candidates[0].content.parts[0].inline_data.data  # raw 24kHz/16-bit PCM

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_wav(out / "ref.wav", pcm)
    (out / "ref.txt").write_text(args.text, encoding="utf-8")
    print(f"wrote {out}/ref.wav + ref.txt  ({len(pcm)} PCM bytes)")
    print("then generate the bank in this accent:")
    print(f"  python -m wordbank --lang <lang> generate --words PHRASES.txt \\")
    print(f"    --ref-audio {out}/ref.wav --ref-text-file {out}/ref.txt --voice <lang>")


if __name__ == "__main__":
    main()
