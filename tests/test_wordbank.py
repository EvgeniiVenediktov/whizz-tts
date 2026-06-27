"""Unit tests for the cloud word-bank pipeline (mocked backend + verifier)."""
import io

import numpy as np
import soundfile as sf

from wordbank.backends.chirp import ChirpTTSBackend
from wordbank.manifest import Manifest, PENDING, VERIFIED, QUARANTINED
from wordbank.pipeline import WordBankPipeline
from wordbank.paths import safe_filename
from wordbank.verify import Verifier, normalize
from wordbank import postproc


def make_wav(freq=160.0, dur=0.5, sr=24_000) -> bytes:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    pcm = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, pcm, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def fake_synth(text, voice, language, ssml):
    return make_wav()


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
    m.set_status(wid, VERIFIED, file_path="/x.ogg", transcript="cat", billed_chars=3)
    assert m.counts().get(VERIFIED) == 1
    assert m.pending() == []           # verified no longer pending
    m.close()


# ---- verify normalization ----

def test_normalize():
    assert normalize("Two,") == "two"
    assert normalize("House.") == "house"
    assert normalize("  HELLO!! ") == "hello"


def test_verifier_with_fake_transcriber():
    v = Verifier(transcribe_fn=lambda pcm, sr: "Cat.")
    ok, t = v.verify(np.zeros(10, np.float32), 24_000, "cat")
    assert ok and t == "Cat."
    ok2, _ = v.verify(np.zeros(10, np.float32), 24_000, "dog")
    assert not ok2


# ---- chirp backend (synth_fn seam) ----

def test_chirp_backend_returns_pcm():
    backend = ChirpTTSBackend(synth_fn=fake_synth)
    res = backend.synthesize("cat", voice="v", language="en-US")
    assert res.sample_rate == 24_000
    assert res.pcm.ndim == 1 and len(res.pcm) > 0
    assert res.billed_chars == 3


# ---- post-processing ----

def test_to_ogg_magic_bytes():
    pcm = (0.3 * np.sin(2 * np.pi * 200 * np.linspace(0, 0.3, 7200))).astype(np.float32)
    assert postproc.to_ogg(pcm, 24_000)[:4] == b"OggS"


def test_safe_filename():
    assert safe_filename("Cat") == "cat"
    assert safe_filename("don't") == "don-t"
    assert safe_filename("!!!") == "word"


# ---- pipeline end-to-end (all mocked) ----

def test_pipeline_success_and_resume(tmp_path):
    m = Manifest(str(tmp_path / "wb.db"))
    m.add_words(["cat", "dog"], "en")
    pipe = WordBankPipeline(
        m, ChirpTTSBackend(synth_fn=fake_synth), PassVerifier(),
        str(tmp_path / "out"), lang="en", voice="testvoice",
    )
    result = pipe.run()
    assert result[VERIFIED] == 2
    assert m.counts().get(VERIFIED) == 2

    # files written where expected
    files = list((tmp_path / "out").rglob("*.ogg"))
    assert len(files) == 2
    assert all(f.read_bytes()[:4] == b"OggS" for f in files)

    # resume: nothing left to do
    assert pipe.run() == {VERIFIED: 0, "failed": 0, QUARANTINED: 0}
    m.close()


def test_pipeline_quarantines_after_max_attempts(tmp_path):
    m = Manifest(str(tmp_path / "wb.db"))
    m.add_words(["cat", "dog"], "en")
    pipe = WordBankPipeline(
        m, ChirpTTSBackend(synth_fn=fake_synth), FailVerifier(),
        str(tmp_path / "out"), lang="en", voice="testvoice", max_attempts=2,
    )
    pipe.run()
    assert m.counts().get(QUARANTINED) == 2
    assert not list((tmp_path / "out").rglob("*.ogg"))   # no files for failed words
    m.close()
