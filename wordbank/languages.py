"""Curated list of languages the generative model can produce, and the shape
every language code must have.

This is a MANUAL gate, not derived from data. **Presence in this list means
verified** — add a language ONLY after generating sample audio for it AND
checking quality by listening. See the README TODO.

Codes are BCP-47 style and must match the bank's `lang` field, e.g. "en-us",
"en-gb", "fr".

Also the curated Gemini voices per language (VOICE_LANGUAGES), a separate fact.
"""
import json
import pathlib
import re

SUPPORTED_LANGUAGES = [
    {"code": "en-us", "name": "English (US)"},
    # Add once generated + verified by ear:
    # {"code": "en-gb", "name": "English (British)"},
    # {"code": "fr",    "name": "French"},
    # {"code": "es",    "name": "Spanish"},
]

# One female + one male Gemini voice per bank `lang` (gemini_voices.json, picked
# by hand; scripts/fetch_gemini_voices.py refreshes their metadata). Voices
# /v1/speak accepts, not a verified bank: unrelated to SUPPORTED_LANGUAGES.
VOICE_LANGUAGES_FILE = pathlib.Path(__file__).with_name("gemini_voices.json")
VOICE_LANGUAGES = json.loads(VOICE_LANGUAGES_FILE.read_text(encoding="utf-8"))["languages"]

# A bank `lang` in canonical form: lowercase BCP-47 style ("fr", "en-us",
# "ru-ru", "zh-cn"). It is also the first directory of every stored clip
# (`<lang>/<voice>/…`), so this shape is what keeps a request's `lang` from
# naming a path outside the bank ("../x", "/abs", "en/../..").
_LANG_CODE = re.compile(r"[a-z]{2,3}(?:-[a-z0-9]{2,8})*")
_LANG_MAX_LEN = 35   # BCP-47's recommended minimum buffer; keeps it one short path segment


def is_lang_code(code) -> bool:
    """True iff `code` is already a canonical bank language code."""
    return (isinstance(code, str) and len(code) <= _LANG_MAX_LEN
            and _LANG_CODE.fullmatch(code) is not None)


def normalize_lang(code: str) -> str:
    """Canonical form of a user-supplied language code: trimmed and lowercased
    (BCP-47 is case-insensitive and the bank stores lowercase, so "EN-US" finds
    the "en-us" rows). Raises ValueError for anything that isn't a code.

    Non-ASCII is refused before lowercasing: some letters fold to ASCII (the
    Kelvin sign lowercases to "k"), which would let a look-alike pass.
    """
    canon = code.strip().lower() if isinstance(code, str) and code.isascii() else None
    if not is_lang_code(canon):
        raise ValueError(f"invalid lang {code!r}: expected a language code like 'en-us'")
    return canon
