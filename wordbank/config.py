import os
from dataclasses import dataclass


@dataclass
class Config:
    db: str = "wordbank.db"
    out_dir: str = "output/words"
    lang: str = "en-us"          # serving/bank code (BCP-47); matches /v1/languages
    backend: str = "chirp"
    voice: str = "en-US-Chirp3-HD-Charon"
    max_attempts: int = 3

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            db=os.getenv("WORDBANK_DB", "wordbank.db"),
            out_dir=os.getenv("WORDBANK_OUT", "output/words"),
            lang=os.getenv("WORDBANK_LANG", "en-us"),
            backend=os.getenv("WORDBANK_BACKEND", "chirp"),
            voice=os.getenv("WORDBANK_VOICE", "en-US-Chirp3-HD-Charon"),
            max_attempts=int(os.getenv("WORDBANK_MAX_ATTEMPTS", "3")),
        )
