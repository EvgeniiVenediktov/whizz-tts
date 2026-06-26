"""List models your key can use and which methods they support.

Answers: is there a TTS model, and does any TTS model support batchGenerateContent?

  GEMINI_API_KEY=… python scripts/list_models.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from google import genai


def main() -> None:
    client = genai.Client()
    models = list(client.models.list())

    print("=== TTS models ===")
    for m in models:
        actions = getattr(m, "supported_actions", None) or []
        if "tts" in m.name.lower() or "audio" in (m.display_name or "").lower():
            print(f"  {m.name}  -> {actions}")

    print("\n=== models supporting batchGenerateContent ===")
    for m in models:
        actions = getattr(m, "supported_actions", None) or []
        if "batchGenerateContent" in actions:
            print(f"  {m.name}  -> {actions}")


if __name__ == "__main__":
    main()
