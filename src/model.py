from typing import Optional

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


def load_model(model_id: str = "k2-fsa/OmniVoice") -> None:
    global _model, _device
    _device = select_device()
    dtype = select_dtype(_device)
    _model = OmniVoice.from_pretrained(model_id, device_map=_device, dtype=dtype)


def get_device() -> str:
    return _device


def is_loaded() -> bool:
    return _model is not None


def ensure_loaded(model_id: str = "k2-fsa/OmniVoice") -> None:
    """Load the model on first use (lazy), so the server can start without it."""
    if _model is None:
        load_model(model_id)


def transcribe(audio: np.ndarray, sample_rate: int = 24_000) -> str:
    """Transcribe audio using OmniVoice's bundled Whisper ASR.

    Lazily loads the ASR model on first call.
    """
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
    if _model is None:
        raise RuntimeError("Model not loaded — call load_model() first")
    # Seeding makes generation reproducible. Without it, the default
    # position_temperature randomizes output run-to-run.
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
