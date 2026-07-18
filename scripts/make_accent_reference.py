"""Generate an accent reference clip with Gemini — the local model's voice-clone
reference. OmniVoice then clones this accent/voice for the whole phrase bank.

Writes <out>/ref.wav + <out>/ref.txt (the spoken phrase, for --ref-text-file).
Needs GEMINI_API_KEY. To synth AND register in one step instead, use the CLI:
`python -m wordbank --lang <lang> gemini-voice --text "…"`.

  GEMINI_API_KEY=… python scripts/make_accent_reference.py \
    --text "The quick brown fox jumps over the lazy dog." \
    --instruct "Speak in a British (Received Pronunciation) accent" \
    --voice Kore --out refs/en-gb
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from wordbank.gemini_batch import write_wav_from_pcm_bytes  # raw PCM → WAV


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True, help="reference phrase to speak")
    p.add_argument("--instruct", default="",
                   help="accent/style direction, e.g. 'Speak in a British accent' (not spoken)")
    p.add_argument("--voice", default="Kore", help="Gemini prebuilt voice")
    p.add_argument("--model", default="gemini-2.5-flash-preview-tts")
    p.add_argument("--out", default="refs/ref", help="output dir for ref.wav + ref.txt")
    args = p.parse_args()

    from wordbank.gemini_tts import synthesize_pcm

    pcm = synthesize_pcm(args.text, voice=args.voice, instruct=args.instruct,
                         model=args.model)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_wav_from_pcm_bytes(out / "ref.wav", pcm)
    (out / "ref.txt").write_text(args.text, encoding="utf-8")
    print(f"wrote {out}/ref.wav + ref.txt  ({len(pcm)} PCM bytes)")
    print("then register the voice and generate the bank in this accent:")
    print(f"  python -m wordbank --lang <lang> add-voice --ref-audio {out}/ref.wav \\")
    print(f"    --ref-text-file {out}/ref.txt --source gemini")
    print(f"  python -m wordbank --lang <lang> generate --words PHRASES.txt --voice <lang>")


if __name__ == "__main__":
    main()
