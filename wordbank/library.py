"""Library description (what a provisioned bank holds: counts per language
and voice, OGG count, folders) and its verification at deploy time. Word
counts are clips (a word in two voices counts twice). Read-only BankLookup +
stdlib, so it runs in the serving image.
"""
import json
import os
import pathlib
from typing import List, Optional

from .lookup import BankLookup


def describe(db: str, ogg_root: str) -> dict:
    """Introspect a bank into a description dict (what a valid volume must hold)."""
    bl = BankLookup(db)
    wb = bl.languages()
    pb = bl.phrase_languages()
    wv = bl.word_voices()
    bl.close()

    ogg = pathlib.Path(ogg_root)
    return {
        "db": os.path.basename(db),
        "folders": ["words_ogg"],
        "words": sum(l["words"] for l in wb),
        "phrases": sum(l["words"] for l in pb),
        "ogg_files": sum(1 for _ in ogg.rglob("*.ogg")) if ogg.is_dir() else 0,
        "languages": {"wordbank": wb, "phrasebank": pb, "wordbank_voices": wv},
    }


def verify(desc_path: str, root: Optional[str] = None, *, db: Optional[str] = None,
           ogg_root: Optional[str] = None) -> List[str]:
    """Problems of a bank against the description ([] = OK). The word count
    must match; phrase and OGG counts are minimums (/v1/speak grows them on a
    live volume). db and ogg_root resolve independently: explicit kwargs, else
    the flat deploy layout under `root` (`root/desc["db"]`, `root/words_ogg`),
    else the server's own defaults ($BANK_DB, $OGG_ROOT)."""
    dp = pathlib.Path(desc_path)
    if not dp.exists():
        return [f"no description file at {desc_path}"]
    desc = json.loads(dp.read_text(encoding="utf-8"))

    root_path = pathlib.Path(root) if root is not None else None
    if db is None:
        db = str(root_path / desc.get("db", "wordbank.db")) if root_path is not None \
            else os.getenv("BANK_DB", "wordbank.db")
    if ogg_root is None:
        ogg_root = str(root_path / "words_ogg") if root_path is not None \
            else os.getenv("OGG_ROOT", "output/words_ogg")

    problems: List[str] = []
    tree = pathlib.Path(ogg_root)
    if not tree.is_dir():
        problems.append(f"missing folder: words_ogg (looked in {tree.resolve()})")

    db_path = pathlib.Path(db)
    bl = BankLookup(db)
    words = sum(l["words"] for l in bl.languages())
    phrases = sum(l["words"] for l in bl.phrase_languages())
    bl.close()
    if "words" in desc and words != desc["words"]:
        problems.append(f"word count {words} != expected {desc['words']} (db: {db_path.resolve()})")
    if "phrases" in desc and phrases < desc["phrases"]:
        problems.append(f"phrase count {phrases} < expected {desc['phrases']} (db: {db_path.resolve()})")
    if "ogg_files" in desc:
        n = sum(1 for _ in tree.rglob("*.ogg")) if tree.is_dir() else 0
        if n < desc["ogg_files"]:
            problems.append(
                f"ogg files {n} < expected {desc['ogg_files']} (looked in {tree.resolve()})")
    return problems
