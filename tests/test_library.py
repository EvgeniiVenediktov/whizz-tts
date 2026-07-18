"""Tests for the deployment library description + verification."""
import json
import shutil

from wordbank.library import describe, verify
from wordbank.manifest import DONE, VERIFIED, Manifest


def _seed_root(root):
    """Build a /data-like tree: wordbank.db + words_ogg/ (2 clips) + voices/."""
    wm = Manifest(str(root / "wordbank.db"))
    pm = Manifest(str(root / "wordbank.db"), table="phrases", text_col="text")
    wm.add_words(["cat", "dog"], "en-us", voice="Kore")
    for r in wm.iter_rows():
        wm.set_status(r["id"], DONE,
                      file_path=f"en-us/gemini-Kore/0/{r['id']}_{r['word']}.wav", voice="Kore")
    pm.add_words(["hello world"], "en-us", voice="en-gb")
    pm.set_status(pm.iter_rows()[0]["id"], VERIFIED,
                  file_path="en-us/local-en-gb/0/1_hello.wav", voice="en-gb")
    wm.close()
    pm.close()

    ogg = root / "words_ogg" / "en-us" / "gemini-Kore" / "0"
    ogg.mkdir(parents=True)
    (ogg / "1_cat.ogg").write_bytes(b"OggS")
    (ogg / "2_dog.ogg").write_bytes(b"OggS")
    (root / "voices" / "en-us").mkdir(parents=True)
    (root / "voices" / "en-us" / "kore.wav").write_bytes(b"RIFF")


def test_describe_captures_bank(tmp_path):
    _seed_root(tmp_path)
    d = describe(str(tmp_path / "wordbank.db"), str(tmp_path / "words_ogg"),
                 str(tmp_path / "voices"))
    assert d["db"] == "wordbank.db"
    assert d["words"] == 2 and d["phrases"] == 1 and d["ogg_files"] == 2
    assert d["folders"] == ["words_ogg", "voices"]
    assert {"code": "en-us", "words": 2} in d["languages"]["wordbank"]
    assert {"code": "en-us", "words": 1} in d["languages"]["phrasebank"]


def test_verify_ok(tmp_path):
    _seed_root(tmp_path)
    d = describe(str(tmp_path / "wordbank.db"), str(tmp_path / "words_ogg"),
                 str(tmp_path / "voices"))
    (tmp_path / "library.json").write_text(json.dumps(d))
    assert verify(str(tmp_path / "library.json"), str(tmp_path)) == []


def test_verify_missing_description(tmp_path):
    problems = verify(str(tmp_path / "nope.json"), str(tmp_path))
    assert len(problems) == 1 and "no description file" in problems[0]


def test_verify_detects_wrong_count_and_missing_folder(tmp_path):
    _seed_root(tmp_path)
    desc = describe(str(tmp_path / "wordbank.db"), str(tmp_path / "words_ogg"),
                    str(tmp_path / "voices"))
    (tmp_path / "library.json").write_text(json.dumps(desc))

    # Drop an OGG and the voices folder → verification should flag both.
    (tmp_path / "words_ogg" / "en-us" / "gemini-Kore" / "0" / "2_dog.ogg").unlink()
    shutil.rmtree(tmp_path / "voices")

    problems = verify(str(tmp_path / "library.json"), str(tmp_path))
    assert any("ogg files" in p for p in problems)
    assert any("missing folder: voices" in p for p in problems)


def test_verify_empty_volume_is_invalid(tmp_path):
    """First deploy: only the description is present → invalid (triggers unpack)."""
    _seed_root(tmp_path)
    desc = describe(str(tmp_path / "wordbank.db"), str(tmp_path / "words_ogg"),
                    str(tmp_path / "voices"))
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "library.json").write_text(json.dumps(desc))
    problems = verify(str(empty / "library.json"), str(empty))
    assert problems  # folders missing + counts wrong
