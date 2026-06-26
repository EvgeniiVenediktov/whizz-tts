"""Verify generated word audio with local Whisper STT.

The transcriber is injectable so unit tests don't need to load Whisper.
"""
import difflib
import re
from typing import Callable, Optional, Tuple

import numpy as np

_NUM2WORD = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
    "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
    "18": "eighteen", "19": "nineteen", "20": "twenty",
}


def normalize(s: str) -> str:
    """Lowercase, strip punctuation, collapse spaces, map digits to words."""
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    tokens = [_NUM2WORD.get(t, t) for t in s.split()]
    return " ".join(tokens)


def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _make_whisper_transcriber() -> Callable[[np.ndarray, int], str]:
    """Lazily build a Whisper ASR transcriber (only for real runs)."""
    from transformers import pipeline as hf_pipeline

    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
    )

    def transcribe(pcm: np.ndarray, sr: int) -> str:
        out = pipe({"array": pcm.astype(np.float32), "sampling_rate": sr})
        return out["text"].strip()

    return transcribe


class Verifier:
    def __init__(self, transcribe_fn: Optional[Callable[[np.ndarray, int], str]] = None,
                 threshold: float = 0.8):
        self._transcribe_fn = transcribe_fn
        self.threshold = threshold

    def _transcriber(self) -> Callable[[np.ndarray, int], str]:
        if self._transcribe_fn is None:
            self._transcribe_fn = _make_whisper_transcriber()
        return self._transcribe_fn

    def verify(self, pcm: np.ndarray, sr: int, expected: str) -> Tuple[bool, str]:
        transcript = self._transcriber()(pcm, sr)
        exp, got = normalize(expected), normalize(transcript)
        ok = bool(exp) and (
            exp == got
            or exp in got.split()          # word appears among transcribed tokens
            or _similar(exp, got) >= self.threshold
        )
        return ok, transcript
