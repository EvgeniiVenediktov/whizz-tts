"""Tests for the word-list cleaner."""
from wordbank.wordlist import clean_words


# ---- word list cleaning ----

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
