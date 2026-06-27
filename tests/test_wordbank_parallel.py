"""Tests for the word-list cleaner, rate limiter, and parallel pipeline run."""
import time

from tests.conftest import FakeBackend
from wordbank.manifest import Manifest, QUARANTINED, VERIFIED
from wordbank.pipeline import WordBankPipeline
from wordbank.ratelimit import RateLimiter
from wordbank.wordlist import clean_words


class PassVerifier:
    def verify(self, pcm, sr, expected):
        return True, expected


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


def test_clean_words_preserves_frequency_order():
    assert clean_words(["zebra", "apple", "mango"]) == ["zebra", "apple", "mango"]


# ---- rate limiter ----

def test_rate_limiter_allows_burst_then_throttles():
    rl = RateLimiter(rate_per_min=60)       # 1 token/sec, capacity 60
    start = time.monotonic()
    for _ in range(60):                     # full bucket → immediate
        rl.acquire()
    assert time.monotonic() - start < 0.5
    rl.acquire()                            # 61st must wait ~1s for a refill
    assert time.monotonic() - start >= 0.9


# ---- parallel pipeline ----

def test_parallel_run_processes_all(tmp_path):
    m = Manifest(str(tmp_path / "wb.db"))
    m.add_words([f"word{i}" for i in range(50)], "en")
    pipe = WordBankPipeline(
        m, FakeBackend(), PassVerifier(), str(tmp_path / "out"),
        lang="en", voice="testvoice", rate_limiter=RateLimiter(100_000),
    )
    result = pipe.run_parallel(workers=8)
    assert result[VERIFIED] == 50
    assert m.counts().get(VERIFIED) == 50
    assert len(list((tmp_path / "out").rglob("*.wav"))) == 50
    # resume: nothing left
    assert pipe.run_parallel(workers=8) == {VERIFIED: 0, "failed": 0, QUARANTINED: 0}
    m.close()
