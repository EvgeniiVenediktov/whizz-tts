"""Curated list of languages the generative model can produce.

This is a MANUAL gate, not derived from data. **Presence in this list means
verified** — add a language ONLY after generating sample audio for it AND
checking quality (incl. voice cloning) by listening. See the README TODO.

Codes are BCP-47 style and must match the bank's `lang` field, e.g. "en-us",
"en-gb", "fr".
"""

SUPPORTED_LANGUAGES = [
    {"code": "en-us", "name": "English (US)"},
    # Add once generated + verified by ear:
    # {"code": "en-gb", "name": "English (British)"},
    # {"code": "fr",    "name": "French"},
    # {"code": "es",    "name": "Spanish"},
]
