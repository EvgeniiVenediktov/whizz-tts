"""Manifest bookkeeping and path helpers."""
from wordbank.manifest import DONE, Manifest
from wordbank.paths import safe_filename, voice_from_path


def test_manifest_add_dedup(tmp_path):
    m = Manifest(str(tmp_path / "wb.db"))
    assert m.add_words(["cat", "dog", "cat"], "en") == 2     # dedup within batch
    assert m.add_words(["dog", "fish"], "en") == 1            # dedup vs existing
    assert len(m.pending()) == 3
    m.close()


def test_manifest_attempts_and_status(tmp_path):
    m = Manifest(str(tmp_path / "wb.db"))
    m.add_words(["cat"], "en")
    wid = m.pending()[0]["id"]
    assert m.start_attempt(wid) == 1
    assert m.start_attempt(wid) == 2
    m.set_status(wid, DONE, file_path="x.wav", billed_chars=3)
    assert m.counts().get(DONE) == 1
    assert m.pending() == []           # done no longer pending
    m.close()


def test_safe_filename():
    assert safe_filename("Cat") == "cat"
    assert safe_filename("don't") == "don-t"
    assert safe_filename("!!!") == "word"


def test_voice_from_path():
    assert voice_from_path("en-us/gemini-Kore/0/1_cat.wav") == "Kore"
    assert voice_from_path("en-gb/local-en-gb/0/1_x.wav") == "en-gb"
    assert voice_from_path("nope.wav") is None

