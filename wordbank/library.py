"""Library description + verification for deployment.

A **description file** (JSON) records what a provisioned bank should contain —
languages, voices, word/phrase counts, OGG file count, and the folders that must
exist. At deploy time the container verifies the mounted volume against it and
unpacks the library archive only if verification fails. Uses read-only
`BankLookup` (never creates/mutates the DB) + the stdlib, so it runs in the
bank-only image with no extra deps.
"""
import json
import os
import pathlib
from typing import List, Optional

from .lookup import BankLookup


def describe(db: str, ogg_root: str, voices_dir: Optional[str] = None) -> dict:
    """Introspect a bank into a description dict (what a valid volume must hold)."""
    bl = BankLookup(db)
    wb = bl.languages()
    pb = bl.phrase_languages()
    voices = [{"name": v["name"], "lang": v["lang"]} for v in bl.voices()]
    bl.close()

    folders = ["words_ogg"]
    if voices_dir and os.path.isdir(voices_dir):
        folders.append("voices")
    ogg = pathlib.Path(ogg_root)
    return {
        "db": os.path.basename(db),
        "folders": folders,
        "words": sum(l["words"] for l in wb),
        "phrases": sum(l["words"] for l in pb),
        "ogg_files": sum(1 for _ in ogg.rglob("*.ogg")) if ogg.is_dir() else 0,
        "languages": {"wordbank": wb, "phrasebank": pb},
        "voices": voices,
    }


def verify(desc_path: str, root: str) -> List[str]:
    """Check `root` against the description. Returns a list of problems (empty = OK)."""
    dp = pathlib.Path(desc_path)
    if not dp.exists():
        return [f"no description file at {desc_path}"]
    desc = json.loads(dp.read_text(encoding="utf-8"))
    root = pathlib.Path(root)

    problems: List[str] = []
    for folder in desc.get("folders", []):
        if not (root / folder).is_dir():
            problems.append(f"missing folder: {folder}")

    bl = BankLookup(str(root / desc.get("db", "wordbank.db")))
    words = sum(l["words"] for l in bl.languages())
    phrases = sum(l["words"] for l in bl.phrase_languages())
    bl.close()
    if "words" in desc and words != desc["words"]:
        problems.append(f"word count {words} != expected {desc['words']}")
    if "phrases" in desc and phrases != desc["phrases"]:
        problems.append(f"phrase count {phrases} != expected {desc['phrases']}")
    if "ogg_files" in desc:
        tree = root / "words_ogg"
        n = sum(1 for _ in tree.rglob("*.ogg")) if tree.is_dir() else 0
        if n != desc["ogg_files"]:
            problems.append(f"ogg files {n} != expected {desc['ogg_files']}")
    return problems
