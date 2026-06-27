"""Unit tests for the sync word-bank pipeline (generic fake backend + verifier)."""
import numpy as np

from tests.conftest import FakeBackend
from wordbank.manifest import Manifest, QUARANTINED, VERIFIED
from wordbank.paths import safe_filename
from wordbank.pipeline import WordBankPipeline
from wordbank.verify import Verifier, normalize


class PassVerifier:
    def verify(self, pcm, sr, expected):
        return True, expected


class FailVerifier:
    def verify(self, pcm, sr, expected):
        return False, "wrong"


# ---- manifest ----

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
    m.set_status(wid, VERIFIED, file_path="x.wav", transcript="cat", billed_chars=3)
    assert m.counts().get(VERIFIED) == 1
    assert m.pending() == []           # verified no longer pending
    m.close()


# ---- verify / helpers ----

def test_normalize():
    assert normalize("Two,") == "two"
    assert normalize("House.") == "house"
    assert normalize("  HELLO!! ") == "hello"


def test_verifier_with_fake_transcriber():
    v = Verifier(transcribe_fn=lambda pcm, sr: "Cat.")
    ok, t = v.verify(np.zeros(10, np.float32), 24_000, "cat")
    assert ok and t == "Cat."
    assert not v.verify(np.zeros(10, np.float32), 24_000, "dog")[0]


def test_safe_filename():
    assert safe_filename("Cat") == "cat"
    assert safe_filename("don't") == "don-t"
    assert safe_filename("!!!") == "word"


# ---- pipeline → WAV masters (all mocked) ----

def test_pipeline_without_verifier_writes(tmp_path):
    """verifier=None → clip is written with no STT (used for bulk local runs)."""
    m = Manifest(str(tmp_path / "wb.db"))
    m.add_words(["cat"], "en-gb")
    pipe = WordBankPipeline(m, FakeBackend(), None, str(tmp_path / "out"),
                            lang="en-gb", voice="en-gb")
    assert pipe.run()[VERIFIED] == 1
    wavs = list((tmp_path / "out").rglob("*.wav"))
    assert len(wavs) == 1 and wavs[0].read_bytes()[:4] == b"RIFF"
    m.close()


def test_pipeline_success_and_resume(tmp_path):
    m = Manifest(str(tmp_path / "wb.db"))
    m.add_words(["cat", "dog"], "en")
    pipe = WordBankPipeline(m, FakeBackend(), PassVerifier(),
                            str(tmp_path / "out"), lang="en", voice="testvoice")
    assert pipe.run()[VERIFIED] == 2
    assert m.counts().get(VERIFIED) == 2
    files = list((tmp_path / "out").rglob("*.wav"))
    assert len(files) == 2 and all(f.read_bytes()[:4] == b"RIFF" for f in files)
    # resume: nothing left to do
    assert pipe.run() == {VERIFIED: 0, "failed": 0, QUARANTINED: 0}
    m.close()


def test_pipeline_quarantines_after_max_attempts(tmp_path):
    m = Manifest(str(tmp_path / "wb.db"))
    m.add_words(["cat", "dog"], "en")
    pipe = WordBankPipeline(m, FakeBackend(), FailVerifier(),
                            str(tmp_path / "out"), lang="en", voice="testvoice",
                            max_attempts=2)
    pipe.run()
    assert m.counts().get(QUARANTINED) == 2
    assert not list((tmp_path / "out").rglob("*.wav"))   # no files for failed words
    m.close()
