"""Storage + path resolution for voice-clone references.

A reference is a small WAV clip plus its transcript. Clips live under
VOICES_ROOT as ``<lang>/<name>.wav``; the manifest `voices` table holds the
metadata (the relative path, ref_text, instruct, source). Generation-side only —
references are never part of the served bank.

No heavy deps (no torch/genai) so the server can import this in the bank-only build.
"""
import os
import pathlib

from .manifest import Manifest
from .paths import safe_filename


def voice_rel_path(lang: str, name: str) -> str:
    """Path of a reference clip relative to VOICES_ROOT: ``<lang>/<safe-name>.wav``."""
    return os.path.join(lang, f"{safe_filename(name)}.wav")


def resolve_voice_path(voices_root: str, ref_audio: str) -> str:
    """Resolve a stored `ref_audio` to a usable filesystem path.

    Order: absolute → as-given if it exists (CLI-registered repo-relative paths
    like ``refs/en-gb/ref.wav``) → under VOICES_ROOT (UI-saved ``<lang>/<name>.wav``).
    """
    if os.path.isabs(ref_audio) or os.path.exists(ref_audio):
        return ref_audio
    return os.path.join(voices_root, ref_audio)


def save_voice_clip(manifest: Manifest, voices_root: str, name: str, lang: str,
                    wav_bytes: bytes, ref_text: str, *, instruct: str = None,
                    source: str = "gemini", sample_rate: int = None) -> str:
    """Write a reference WAV under VOICES_ROOT and register it in the manifest.

    Returns the path stored in the table (relative to VOICES_ROOT).
    """
    rel = voice_rel_path(lang, name)
    path = pathlib.Path(voices_root) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(wav_bytes)
    manifest.add_voice(name, lang, rel, ref_text=ref_text, instruct=instruct,
                       source=source, sample_rate=sample_rate)
    return rel
