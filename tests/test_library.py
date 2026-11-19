"""Tests for the deployment library description + verification."""
import json
import pathlib
import shutil

from wordbank.library import describe, verify
from wordbank.manifest import DONE, Manifest


def _seed_root(root, db=None, ogg_root=None):
    """Build a /data-like tree: wordbank.db + words_ogg/ (2 clips).

    By default everything lives flat under `root` (the deploy layout). Pass
    `db`/`ogg_root` to place them at different paths (dev layout).
    """
    db = pathlib.Path(db) if db is not None else root / "wordbank.db"
    ogg_root = pathlib.Path(ogg_root) if ogg_root is not None else root / "words_ogg"

    wm = Manifest(str(db))
    pm = Manifest(str(db), table="phrases", text_col="text")
    wm.add_words(["cat", "dog"], "en-us", voice="Kore")
    for r in wm.iter_rows():
        wm.set_status(r["id"], DONE,
                      file_path=f"en-us/gemini-Kore/0/{r['id']}_{r['word']}.wav", voice="Kore")
    pm.add_words(["hello world"], "en-us", voice="en-gb-tutor-11")
    pm.set_status(pm.iter_rows()[0]["id"], DONE,
                  file_path="en-us/gemini-en-gb-tutor-11/phrases/0/1_hello.wav", voice="en-gb-tutor-11")
    wm.close()
    pm.close()

    ogg = ogg_root / "en-us" / "gemini-Kore" / "0"
    ogg.mkdir(parents=True)
    (ogg / "1_cat.ogg").write_bytes(b"OggS")
    (ogg / "2_dog.ogg").write_bytes(b"OggS")


def _describe(root):
    desc = describe(str(root / "wordbank.db"), str(root / "words_ogg"))
    (root / "library.json").write_text(json.dumps(desc))
    return desc


def test_describe_captures_bank(tmp_path):
    _seed_root(tmp_path)
    d = describe(str(tmp_path / "wordbank.db"), str(tmp_path / "words_ogg"))
    assert d["db"] == "wordbank.db"
    assert d["words"] == 2 and d["phrases"] == 1 and d["ogg_files"] == 2
    assert d["folders"] == ["words_ogg"]
    assert {"code": "en-us", "words": 2} in d["languages"]["wordbank"]
    assert {"code": "en-us", "words": 1} in d["languages"]["phrasebank"]


def test_describe_counts_clips_per_voice(tmp_path):
    """A word banked in two voices is two clips; the breakdown says which."""
    _seed_root(tmp_path)                                      # cat, dog in Kore
    wm = Manifest(str(tmp_path / "wordbank.db"))
    wm.add_words(["cat"], "en-us", voice="Iapetus")
    rel = "en-us/gemini-Iapetus/0/3_cat.wav"
    wm.set_status(wm.get_entry("cat", "en-us", "Iapetus")["id"], DONE, file_path=rel)
    wm.close()
    (tmp_path / "words_ogg" / "en-us" / "gemini-Iapetus" / "0").mkdir(parents=True)
    (tmp_path / "words_ogg" / (rel[:-4] + ".ogg")).write_bytes(b"OggS")

    desc = _describe(tmp_path)
    assert (desc["words"], desc["ogg_files"]) == (3, 3)
    assert desc["languages"]["wordbank_voices"] == [
        {"code": "en-us", "voice": "Iapetus", "words": 1},
        {"code": "en-us", "voice": "Kore", "words": 2}]
    assert verify(str(tmp_path / "library.json"), str(tmp_path)) == []


def test_verify_missing_description(tmp_path):
    problems = verify(str(tmp_path / "nope.json"), str(tmp_path))
    assert len(problems) == 1 and "no description file" in problems[0]


def test_verify_detects_missing_ogg_and_missing_folder(tmp_path):
    _seed_root(tmp_path)
    _describe(tmp_path)
    (tmp_path / "words_ogg" / "en-us" / "gemini-Kore" / "0" / "2_dog.ogg").unlink()
    problems = verify(str(tmp_path / "library.json"), str(tmp_path))
    assert any("ogg files 1 < expected 2" in p for p in problems)

    shutil.rmtree(tmp_path / "words_ogg")
    problems = verify(str(tmp_path / "library.json"), str(tmp_path))
    missing = next(p for p in problems if "missing folder: words_ogg" in p)
    assert str(tmp_path / "words_ogg") in missing          # names where it looked


def test_verify_ignores_folders_nothing_serves(tmp_path):
    """Older descriptions also list a `voices` folder: not required."""
    _seed_root(tmp_path)
    desc = _describe(tmp_path)
    (tmp_path / "library.json").write_text(json.dumps({**desc, "folders": ["words_ogg", "voices"]}))
    assert verify(str(tmp_path / "library.json"), str(tmp_path)) == []


def test_verify_dev_layout_with_split_roots(tmp_path):
    """Dev layout: db at the root, OGG under output/ — no single --root works,
    but independently-resolved db/ogg_root does."""
    db = tmp_path / "wordbank.db"
    ogg_root = tmp_path / "output" / "words_ogg"
    _seed_root(tmp_path, db=db, ogg_root=ogg_root)
    (tmp_path / "library.json").write_text(json.dumps(describe(str(db), str(ogg_root))))
    assert verify(str(tmp_path / "library.json"), db=str(db), ogg_root=str(ogg_root)) == []


def test_verify_explicit_kwargs_beat_root(tmp_path):
    """Explicit db=/ogg_root= win over --root even when both are given."""
    _seed_root(tmp_path)
    _describe(tmp_path)
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    problems = verify(str(tmp_path / "library.json"), str(empty_root),
                      db=str(tmp_path / "wordbank.db"), ogg_root=str(tmp_path / "words_ogg"))
    assert problems == []


def test_verify_accepts_a_bank_grown_at_runtime(tmp_path):
    """/v1/speak adds phrases + OGGs to the live volume. A restart must still
    verify, or the entrypoint re-unpacks the archive over the grown DB."""
    _seed_root(tmp_path)
    _describe(tmp_path)
    db = str(tmp_path / "wordbank.db")
    assert verify(str(tmp_path / "library.json"), str(tmp_path)) == []

    pm = Manifest(db, table="phrases", text_col="text")
    pm.add_words(["good morning"], "en-us", voice="Puck")
    rel = "en-us/gemini-Puck/phrases/0/2_good-morning.wav"
    pm.set_status(pm.get_entry("good morning", "en-us", "Puck")["id"], DONE,
                  file_path=rel, voice="Puck")
    pm.close()
    ogg = tmp_path / "words_ogg" / (rel[:-4] + ".ogg")
    ogg.parent.mkdir(parents=True)
    ogg.write_bytes(b"OggS")
    assert verify(str(tmp_path / "library.json"), str(tmp_path)) == []

    # Words are never added at runtime, so a word-count change is still drift.
    wm = Manifest(db)
    wm.add_words(["emu"], "en-us")
    wm.set_status(wm.get_entry("emu", "en-us")["id"], DONE, file_path="en-us/x/0/9_emu.wav")
    wm.close()
    assert any("word count" in p for p in verify(str(tmp_path / "library.json"), str(tmp_path)))
