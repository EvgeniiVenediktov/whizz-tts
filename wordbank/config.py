import os
from dataclasses import dataclass


@dataclass
class Config:
    db: str = "wordbank.db"
    out_dir: str = "output/words"
    lang: str = "en-us"          # serving/bank code (BCP-47); matches /v1/languages

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            db=os.getenv("WORDBANK_DB", "wordbank.db"),
            out_dir=os.getenv("WORDBANK_OUT", "output/words"),
            lang=os.getenv("WORDBANK_LANG", "en-us"),
        )
