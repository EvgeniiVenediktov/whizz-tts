from typing import Optional
import threading

import numpy as np
import torch
from omnivoice import OmniVoice

from .device import select_device, select_dtype


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available() and hasattr(torch, "mps"):
        torch.mps.manual_seed(seed)

_model: OmniVoice | None = None
_device: str = "cpu"

# OmniVoice isn't thread-safe, but src.server offloads /v1/tts synthesis to
# asyncio's default (multi-threaded) executor — so concurrent requests can
# call into _model from multiple threads at once. This lock serializes all
# access to _model, and critically covers seeding + generate() as one atomic
# unit: seeding sets torch's *global* RNG, so if two threads could interleave
# seed/generate, one request could silently run with another's seed and break
# the reproducibility guarantee the phrase bank relies on. RLock so guarded
# functions can call each other without self-deadlocking.
_MODEL_LOCK = threading.RLock()


def load_model(model_id: str = "k2-fsa/OmniVoice") -> None:
    global _model, _device
    with _MODEL_LOCK:
        _device = select_device()
        dtype = select_dtype(_device)
        _model = OmniVoice.from_pretrained(model_id, device_map=_device, dtype=dtype)


def get_device() -> str:
    return _device


def is_loaded() -> bool:
    return _model is not None


def ensure_loaded(model_id: str = "k2-fsa/OmniVoice") -> None:
    """Load the model on first use (lazy), so the server can start without it."""
    with _MODEL_LOCK:
        # Guarded to avoid a check-then-act race: two concurrent callers could
        # both observe _model is None and both trigger an (expensive) load,
        # each clobbering the other's model. A slow first load blocking other
        # callers here is intentional, correct behavior.
        if _model is None:
            load_model(model_id)


def transcribe(audio: np.ndarray, sample_rate: int = 24_000) -> str:
    """Transcribe audio using OmniVoice's bundled Whisper ASR.

    Lazily loads the ASR model on first call.
    """
    with _MODEL_LOCK:
        if _model is None:
            raise RuntimeError("Model not loaded — call load_model() first")
        if getattr(_model, "_asr_pipe", None) is None:
            _model.load_asr_model()
        return _model.transcribe((audio, sample_rate))


def generate(
    text: str,
    ref_audio_path: Optional[str] = None,
    ref_text: Optional[str] = None,
    instruct: Optional[str] = None,
    guidance_scale: Optional[float] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    with _MODEL_LOCK:
        if _model is None:
            raise RuntimeError("Model not loaded — call load_model() first")
        # Seeding makes generation reproducible. Without it, the default
        # position_temperature randomizes output run-to-run. Seeding sets
        # torch's global RNG, so it must stay atomic with generate() under
        # the same lock — otherwise a concurrent caller's seed can win and
        # this call silently generates with the wrong seed.
        if seed is not None:
            _seed_everything(seed)
        kwargs = {"text": text}
        if ref_audio_path is not None:
            kwargs["ref_audio"] = ref_audio_path
        if ref_text is not None:
            kwargs["ref_text"] = ref_text
        if instruct is not None:
            kwargs["instruct"] = instruct
        if guidance_scale is not None:
            kwargs["guidance_scale"] = guidance_scale
        result = _model.generate(**kwargs)
        # OmniVoice returns a list of arrays; take the first
        return result[0]
