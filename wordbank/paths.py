"""Map a file under one root to the mirrored path under another root.

Used to keep WAV masters and their OGG copies in parallel trees:
  output/words/en/gemini-Kore/0/1_the.wav
  output/words_ogg/en/gemini-Kore/0/1_the.ogg
"""
import os
import re
from typing import Optional

_SLUG = re.compile(r"[^a-z0-9]+")


def safe_filename(word: str) -> str:
    """Filesystem-safe slug for a word: lowercase, non-alphanumerics → '-'."""
    return _SLUG.sub("-", word.lower()).strip("-") or "word"


def mirror_path(src_path: str, src_root: str, dest_root: str,
                new_ext: str) -> Optional[str]:
    """Return src_path relocated under dest_root with new_ext, or None if
    src_path is not under src_root."""
    rel = os.path.relpath(src_path, src_root)
    if rel.startswith(".."):
        return None
    return os.path.join(dest_root, os.path.splitext(rel)[0] + new_ext)
