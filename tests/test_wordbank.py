"""Manifest bookkeeping, path helpers and the word-list cleaner."""
from wordbank.manifest import DONE, Manifest
from wordbank.paths import safe_filename, voice_from_path
from wordbank.wordlist import clean_words


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
    assert voice_from_path("en-gb/gemini-en-gb-tutor-11/0/1_x.wav") == "en-gb-tutor-11"
    assert voice_from_path("nope.wav") is None


def test_clean_words_filters_and_dedupes():
    raw = ["The", "the", "a", "I", "x", "hello!", "don't", "12", "supercalifragilistic" * 2]
    out = clean_words(raw, min_len=2, max_len=20, alpha_only=True)
    assert "the" in out
    assert out.count("the") == 1            # deduped (case-folded)
    assert "a" in out and "i" in out        # short keepers
    assert "x" not in out                   # too short, not a keeper
    assert "hello" not in out               # punctuation dropped (not alpha)
    assert "don't" not in out               # contraction dropped
    assert "12" not in out                  # numeric dropped
    assert all(2 <= len(w) <= 20 or w in ("a", "i") for w in out)
