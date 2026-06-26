"""Validate the core word-bank mechanism:
  batch of period-separated words -> Whisper word timestamps -> cut -> per-clip STT.

Tests whether the approach is viable before building the full pipeline.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import soundfile as sf
import torch

from omnivoice import OmniVoice
from src.device import select_device, select_dtype

SR = 24_000
WORDS = ["cat", "elephant", "two", "house", "rhythm", "tree", "book", "umbrella"]
OUT = pathlib.Path(__file__).parent.parent / "output" / "word_validation"


def main():
    device = select_device()
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=select_dtype(device))
    model.load_asr_model()

    text = ". ".join(WORDS) + "."
    print(f"Generating batch: {text!r}\n")
    torch.manual_seed(42)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(42)
    audio = model.generate(text=text, instruct="female")[0]
    print(f"batch audio: {len(audio)/SR:.2f}s\n")

    OUT.mkdir(parents=True, exist_ok=True)
    sf.write(OUT / "_batch.wav", audio, SR)

    # Word-level timestamps from the bundled Whisper pipeline
    result = model._asr_pipe(
        {"array": audio.astype(np.float32), "sampling_rate": SR},
        return_timestamps="word",
    )
    chunks = result.get("chunks", [])
    print(f"Whisper found {len(chunks)} word chunks vs {len(WORDS)} input words\n")

    norm = lambda s: "".join(c for c in s.lower() if c.isalnum())
    for i, ch in enumerate(chunks):
        word = ch["text"].strip()
        ts = ch.get("timestamp", (None, None))
        if None in ts:
            print(f"  [{i}] {word!r:14} ts={ts}  (no timestamp)")
            continue
        s, e = ts
        clip = audio[int(s * SR): int(e * SR)]
        if len(clip) < 100:
            print(f"  [{i}] {word!r:14} ts=({s:.2f},{e:.2f})  EMPTY CLIP")
            continue
        # Re-transcribe the ISOLATED clip
        iso = model._asr_pipe({"array": clip.astype(np.float32), "sampling_rate": SR})["text"].strip()
        expected = WORDS[i] if i < len(WORDS) else "?"
        match = "OK" if norm(iso) == norm(expected) else "MISMATCH"
        sf.write(OUT / f"{i:02d}_{norm(expected)}.wav", clip, SR)
        print(f"  [{i}] expected={expected!r:12} batch_word={word!r:12} isolated_stt={iso!r:14} {match}")


if __name__ == "__main__":
    main()
