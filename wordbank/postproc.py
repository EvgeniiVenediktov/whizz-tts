"""Per-clip audio post-processing: trim silence, normalize level, encode OGG."""
import numpy as np


def trim_silence(pcm: np.ndarray, sr: int, top_db: int = 30) -> np.ndarray:
    import librosa

    trimmed, _ = librosa.effects.trim(pcm, top_db=top_db)
    return trimmed if len(trimmed) else pcm


def peak_normalize(pcm: np.ndarray, peak: float = 0.97) -> np.ndarray:
    """Simple peak normalization. (Phase 2: switch to LUFS via pyloudnorm.)"""
    m = float(np.abs(pcm).max()) if len(pcm) else 0.0
    if m < 1e-6:
        return pcm
    return pcm * (peak / m)


def to_ogg(pcm: np.ndarray, sr: int) -> bytes:
    from src.audio import pcm_to_ogg

    return pcm_to_ogg(pcm, sample_rate=sr)
