"""Tests for the word-list cleaner and rate limiter."""
import time

from wordbank.ratelimit import RateLimiter
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


# ---- rate limiter ----

def test_rate_limiter_allows_burst_then_throttles():
    rl = RateLimiter(rate_per_min=60)       # 1 token/sec, capacity 60
    start = time.monotonic()
    for _ in range(60):                     # full bucket → immediate
        rl.acquire()
    assert time.monotonic() - start < 0.5
    rl.acquire()                            # 61st must wait ~1s for a refill
    assert time.monotonic() - start >= 0.9
