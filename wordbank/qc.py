"""End-of-clip QC for single-word clips; stdlib only (the serving image has
no numpy). A clip is flagged when its last END_WINDOW_S is louder than
END_DB_MAX (RMS, dBFS: it ends on sound, usually cut off) or its length is
outside MIN_SECONDS..MAX_SECONDS (a dropped word, or extra speech).
Generation re-takes a flagged clip up to DEFAULT_RETRIES times and otherwise
keeps the best take (best_take).
"""
import array
import math
import pathlib
import sys
import wave
from dataclasses import dataclass
from typing import Optional, Tuple

from .constants import SAMPLE_RATE

END_WINDOW_S = 0.1       # the tail measured
END_DB_MAX = -40.0       # a tail louder than this "ends on sound"
MIN_SECONDS = 0.25
MAX_SECONDS = 3.0
DEFAULT_RETRIES = 2      # extra takes for a flagged clip (CLI --qc-retries)


@dataclass(frozen=True)
class Check:
    seconds: float
    end_db: float

    @property
    def length_ok(self) -> bool:
        return MIN_SECONDS <= self.seconds <= MAX_SECONDS

    @property
    def reasons(self) -> list:
        out = []
        if self.end_db > END_DB_MAX:
            out.append(f"ends on sound ({self.end_db:.0f} dB)")
        if self.seconds < MIN_SECONDS:
            out.append(f"too short ({self.seconds:.2f} s)")
        if self.seconds > MAX_SECONDS:
            out.append(f"too long ({self.seconds:.2f} s)")
        return out

    @property
    def flagged(self) -> bool:
        return bool(self.reasons)

    @property
    def review(self) -> Optional[str]:
        """The `words.review` value: the reasons, or None for a clip that passes."""
        return "; ".join(self.reasons) or None

    @property
    def rank(self) -> Tuple[bool, float]:
        """Sort key, lower is better: a plausible length first (a wrong length
        usually means the wrong content), then the quieter end (least cut off).
        A passing take always ranks first."""
        return (not self.length_ok, self.end_db)


def _check(frames: int, tail: bytes, rate: int) -> Check:
    samples = array.array("h")
    samples.frombytes(tail)
    if sys.byteorder == "big":
        samples.byteswap()                  # WAV/PCM data is little-endian
    if not samples:
        samples = array.array("h", [0])
    rms = math.sqrt(sum(v * v for v in samples) / len(samples))
    # floored at 1e-9 (-180 dB) for digital silence
    return Check(seconds=frames / rate, end_db=20 * math.log10(max(rms / 32768, 1e-9)))


def check_pcm(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> Check:
    """QC raw 16-bit mono PCM (what Gemini returns, via pcm_from_payload)."""
    frames = len(pcm) // 2
    start = max(0, frames - int(sample_rate * END_WINDOW_S))
    return _check(frames, pcm[start * 2:frames * 2], sample_rate)


def check_wav(path) -> Check:
    """QC a 16-bit mono WAV, reading only its header and tail."""
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2 or wf.getnchannels() != 1:
            raise ValueError(f"{path}: not 16-bit mono "
                             f"({wf.getsampwidth() * 8}-bit, {wf.getnchannels()} ch)")
        rate, frames = wf.getframerate(), wf.getnframes()
        window = int(rate * END_WINDOW_S)
        wf.setpos(max(0, frames - window))
        tail = wf.readframes(window)
    return _check(frames, tail, rate)


def read_pcm(path) -> bytes:
    """A bank WAV's frames (16-bit mono PCM), to re-file or compare a take."""
    with wave.open(str(path), "rb") as wf:
        return wf.readframes(wf.getnframes())


def best_take(*takes):
    """The best of (pcm, Check) takes by Check.rank; the earlier one on a tie."""
    return min(takes, key=lambda t: t[1].rank)


def retake_path(db_path: str, row_id: int) -> pathlib.Path:
    """Where a batch row queued for a QC re-take (status `retake`) keeps its
    best take so far: <db dir>/takes/qc/<db name>/<id>.wav, outside the bank
    tree, so scripts/encode_wavs.py never encodes a take that may still be
    replaced (the db name keeps two banks in one directory apart)."""
    db = pathlib.Path(db_path).resolve()
    return db.parent / "takes" / "qc" / db.stem / f"{row_id}.wav"
