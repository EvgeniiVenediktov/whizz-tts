"""Tests for the word-list cleaner, rate limiter, and parallel pipeline run."""
import io
import time

import numpy as np
import soundfile as sf

from wordbank.backends.chirp import ChirpTTSBackend
from wordbank.manifest import Manifest, VERIFIED, QUARANTINED
from wordbank.pipeline import WordBankPipeline
from wordbank.ratelimit import RateLimiter
from wordbank.wordlist import clean_words


def make_wav(sr=24_000, dur=0.3) -> bytes:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    pcm = (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, pcm, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def fake_synth(text, voice, language, ssml):
    return make_wav()


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
    words = [f"word{i}" for i in range(50)]
    m.add_words(words, "en")
    pipe = WordBankPipeline(
        m, ChirpTTSBackend(synth_fn=fake_synth), PassVerifier(),
        str(tmp_path / "out"), lang="en", voice="testvoice",
        rate_limiter=RateLimiter(100_000),   # effectively unthrottled
    )
    result = pipe.run_parallel(workers=8)
    assert result[VERIFIED] == 50
    assert m.counts().get(VERIFIED) == 50
    assert len(list((tmp_path / "out").rglob("*.ogg"))) == 50
    # resume: nothing left
    assert pipe.run_parallel(workers=8) == {VERIFIED: 0, "failed": 0, QUARANTINED: 0}
    m.close()
