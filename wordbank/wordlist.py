"""Pure helpers for cleaning a frequency word list (no I/O, easily testable)."""
import re
from typing import Iterable, List

_ALPHA = re.compile(r"^[a-z]+$")


def clean_words(words: Iterable[str], *, min_len: int = 2, max_len: int = 30,
                alpha_only: bool = True, keep: Iterable[str] = ("a", "i")) -> List[str]:
    """Lowercase, filter, and dedupe a frequency-ordered word list.

    - drops words outside [min_len, max_len]  (except short words in `keep`)
    - alpha_only: keep only a-z words (drops numbers, punctuation, contractions)
    - preserves input (frequency) order; removes duplicates
    """
    keep_set = set(keep)
    seen, out = set(), []
    for raw in words:
        w = raw.strip().lower()
        if not w or w in seen:
            continue
        if w not in keep_set:
            if len(w) < min_len or len(w) > max_len:
                continue
            if alpha_only and not _ALPHA.match(w):
                continue
        seen.add(w)
        out.append(w)
    return out
