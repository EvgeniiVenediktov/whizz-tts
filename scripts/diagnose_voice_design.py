"""Objective diagnostic for voice-design attribute control.

Measures whether gender/accent instructs actually change the output, by:
  1. Sweeping guidance_scale and measuring median F0 (pitch) for male vs female.
     Expectation: female F0 (~165-255 Hz) > male F0 (~85-180 Hz).
  2. Checking whether seeding makes generation deterministic.

Run:  .venv/bin/python scripts/diagnose_voice_design.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import librosa
import torch

from omnivoice import OmniVoice
from src.device import select_device, select_dtype

TEXT = "The quick brown fox jumps over the lazy dog."
SR = 24_000


def median_f0(audio: np.ndarray) -> float:
    f0, voiced, _ = librosa.pyin(
        audio.astype(np.float32), fmin=65, fmax=400, sr=SR
    )
    f0 = f0[~np.isnan(f0)]
    return float(np.median(f0)) if len(f0) else float("nan")


def seeded_generate(model, seed=None, **kwargs):
    if seed is not None:
        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    return model.generate(text=TEXT, **kwargs)[0]


def main():
    device = select_device()
    dtype = select_dtype(device)
    print(f"device={device} dtype={dtype}\n")
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=dtype)

    # --- 1. Determinism check ---
    a = seeded_generate(model, seed=42, instruct="female")
    b = seeded_generate(model, seed=42, instruct="female")
    c = seeded_generate(model, seed=99, instruct="female")
    same = a.shape == b.shape and np.allclose(a, b, atol=1e-4)
    diff = a.shape != c.shape or not np.allclose(a, c[: len(a)] if len(c) >= len(a) else c, atol=1e-4)
    print(f"DETERMINISM: same-seed identical={same}  diff-seed different={diff}\n")

    # --- 2. Guidance sweep: male vs female F0 ---
    REPEATS = 3
    print(f"{'guidance':>9} | {'female F0 (Hz)':>22} | {'male F0 (Hz)':>22} | direction")
    print("-" * 80)
    for g in [1.0, 2.0, 4.0, 7.0]:
        f_f0, m_f0 = [], []
        for i in range(REPEATS):
            af = seeded_generate(model, seed=100 + i, instruct="female", guidance_scale=g)
            am = seeded_generate(model, seed=100 + i, instruct="male", guidance_scale=g)
            f_f0.append(median_f0(af))
            m_f0.append(median_f0(am))
        fm, mm = np.nanmedian(f_f0), np.nanmedian(m_f0)
        f_str = f"{fm:6.1f} ({min(f_f0):.0f}-{max(f_f0):.0f})"
        m_str = f"{mm:6.1f} ({min(m_f0):.0f}-{max(m_f0):.0f})"
        direction = "OK (F>M)" if fm > mm else "REVERSED/FLAT"
        print(f"{g:>9} | {f_str:>22} | {m_str:>22} | {direction}")


if __name__ == "__main__":
    main()
