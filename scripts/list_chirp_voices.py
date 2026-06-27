"""List Cloud Text-to-Speech voices for a locale (Chirp 3 HD only by default).

Use to pick a voice for an accent bank, e.g. en-GB / en-US. Voice names follow
`<locale>-Chirp3-HD-<Name>`. Needs GOOGLE_APPLICATION_CREDENTIALS (a GCP service
account with the Text-to-Speech API) — NOT the Gemini key.

  python scripts/list_chirp_voices.py en-GB
  python scripts/list_chirp_voices.py en-US --all
"""
import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("language", nargs="?", default="en-GB",
                   help="BCP-47 language code (default en-GB)")
    p.add_argument("--all", action="store_true",
                   help="show all voices, not just Chirp 3 HD")
    args = p.parse_args()

    from google.cloud import texttospeech
    client = texttospeech.TextToSpeechClient()
    voices = client.list_voices(language_code=args.language).voices

    n = 0
    for v in sorted(voices, key=lambda v: v.name):
        if args.all or "Chirp3-HD" in v.name:
            print(f"{v.name:<32} {v.ssml_gender.name:<8} {list(v.language_codes)}")
            n += 1
    print(f"\n{n} voice(s) for {args.language}"
          + ("" if args.all else " (Chirp 3 HD; pass --all for the rest)"))


if __name__ == "__main__":
    main()
