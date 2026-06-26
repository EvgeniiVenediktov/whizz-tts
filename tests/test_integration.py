"""
Integration tests — load the real OmniVoice model and call the endpoint.

These are skipped automatically unless the `--integration` flag is passed:
    pytest --integration tests/test_integration.py

Requires:
  - omnivoice installed
  - enough RAM/VRAM for the 0.6B model
  - optionally HF_TOKEN in environment for model download
"""
import pathlib
import re

import pytest
from fastapi.testclient import TestClient

OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "output"

from tests.conftest import make_wav_bytes


def _words(text: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def word_recall(expected: str, actual: str) -> float:
    """Fraction of expected words present in the actual transcription."""
    exp = _words(expected)
    if not exp:
        return 0.0
    return len(exp & _words(actual)) / len(exp)



@pytest.fixture(scope="module")
def real_client():
    from src.server import app
    with TestClient(app) as c:
        yield c


@pytest.mark.integration
def test_health_real(real_client):
    r = real_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.integration
def test_tts_no_ref_real(real_client):
    r = real_client.post("/v1/tts", data={"text": "Hello, this is a test."})
    assert r.status_code == 200
    assert r.content[:4] == b"OggS"


@pytest.mark.integration
def test_tts_generates_sentence(real_client):
    r = real_client.post(
        "/v1/tts",
        data={"text": "The quick brown fox jumps over the lazy dog."},
    )
    assert r.status_code == 200
    assert r.content[:4] == b"OggS"
    assert len(r.content) > 1000

    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / "sentence.ogg"
    out.write_bytes(r.content)
    print(f"\nSaved: {out}")


@pytest.mark.integration
def test_tts_voice_cloning_real(real_client):
    wav = make_wav_bytes(duration_s=3.0)
    r = real_client.post(
        "/v1/tts",
        data={"text": "Hello, this is a test.", "ref_text": "Hello, this is a test."},
        files={"ref_audio": ("ref.wav", wav, "audio/wav")},
    )
    assert r.status_code == 200
    assert r.content[:4] == b"OggS"


# ---------------------------------------------------------------------------
# ASR round-trip: generate speech, transcribe it, verify the words come back.
# Operates at the model level so a failure isolates generation (not OGG).
# ---------------------------------------------------------------------------

SENTENCE = "The quick brown fox jumps over the lazy dog."
MIN_RECALL = 0.6


@pytest.fixture(scope="module")
def loaded_model():
    from src import model
    model.load_model()
    return model


@pytest.mark.integration
@pytest.mark.parametrize(
    "mode,kwargs",
    [
        ("voice_design_female", {"instruct": "female"}),
        ("voice_design_male", {"instruct": "male"}),
        ("auto", {}),
    ],
)
def test_asr_roundtrip(loaded_model, mode, kwargs):
    audio = loaded_model.generate(SENTENCE, **kwargs)
    transcript = loaded_model.transcribe(audio, 24_000)
    recall = word_recall(SENTENCE, transcript)
    print(f"\n[{mode}] transcript: {transcript!r}  recall={recall:.2f}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    import soundfile as sf
    sf.write(OUTPUT_DIR / f"asr_{mode}.wav", audio, 24_000)

    assert recall >= MIN_RECALL, (
        f"[{mode}] generated audio did not transcribe back to the input.\n"
        f"  expected: {SENTENCE!r}\n"
        f"  got:      {transcript!r}\n"
        f"  recall:   {recall:.2f} (threshold {MIN_RECALL})"
    )


# ---------------------------------------------------------------------------
# Objective voice-design attribute checks (no subjective listening).
# ---------------------------------------------------------------------------

def median_f0(audio, sr=24_000) -> float:
    """Median fundamental frequency (pitch) of voiced frames, in Hz."""
    import librosa
    import numpy as np
    f0, _, _ = librosa.pyin(audio.astype(np.float32), fmin=65, fmax=400, sr=sr)
    f0 = f0[~np.isnan(f0)]
    return float(np.median(f0)) if len(f0) else float("nan")


@pytest.mark.integration
def test_seeding_is_deterministic(loaded_model):
    """Same seed → identical audio; different seed → different audio."""
    import numpy as np
    a = loaded_model.generate(SENTENCE, instruct="female", seed=42)
    b = loaded_model.generate(SENTENCE, instruct="female", seed=42)
    c = loaded_model.generate(SENTENCE, instruct="female", seed=7)
    assert a.shape == b.shape and np.allclose(a, b, atol=1e-4), "same seed not reproducible"
    assert a.shape != c.shape or not np.allclose(a, c, atol=1e-4), "different seed gave identical output"


@pytest.mark.integration
def test_gender_pitch_direction(loaded_model):
    """
    Female voice design should produce higher pitch than male on full-sentence
    text. Median over several seeds to average out generation randomness.

    NOTE: this is unreliable on very short text — a documented model limitation,
    so this test deliberately uses a full sentence.
    """
    import numpy as np
    seeds = [100, 101, 102]
    female = np.nanmedian([median_f0(loaded_model.generate(SENTENCE, instruct="female", seed=s)) for s in seeds])
    male = np.nanmedian([median_f0(loaded_model.generate(SENTENCE, instruct="male", seed=s)) for s in seeds])
    print(f"\nfemale median F0={female:.1f}Hz  male median F0={male:.1f}Hz")
    assert female > male * 1.1, (
        f"gender control not working: female F0 {female:.1f}Hz should exceed "
        f"male F0 {male:.1f}Hz by >10%"
    )
