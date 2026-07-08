from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


@dataclass
class SynthResult:
    pcm: np.ndarray       # float32 mono in [-1, 1]
    sample_rate: int
    billed_chars: int     # characters (or SSML length) billed for this call
    seed: Optional[int] = None  # RNG seed used (local model) — for reproducibility


class TTSBackend(Protocol):
    """A cloud (or local) text-to-speech backend.

    Returns PCM so the pipeline can trim/normalize before encoding to OGG.
    """
    name: str

    def synthesize(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        ssml: Optional[str] = None,
    ) -> SynthResult: ...
