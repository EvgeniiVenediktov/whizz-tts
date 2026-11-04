import io
import sys
import types
import wave
import zlib

import numpy as np
import pytest
from fastapi.testclient import TestClient


def pytest_addoption(parser):
    parser.addoption("--integration", action="store_true", default=False)


def pytest_collection_modifyitems(config, items):
    if config.getoption("--integration"):
        return
    skip = pytest.mark.skip(reason="pass --integration to run")
    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(skip)


def make_wav_bytes(duration_s: float = 1.0, sample_rate: int = 24_000) -> bytes:
    """Generate a minimal valid WAV file (440 Hz sine) for use as ref_audio."""
    n_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    samples = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def noise_pcm16(duration_s: float = 1.0, sample_rate: int = 24_000, seed: int = 0) -> bytes:
    """White noise as 16-bit mono PCM bytes (Gemini's output format). Noise, not
    a tone, so Opus actually spends its bitrate and file size reflects it."""
    rng = np.random.RandomState(seed)
    return (rng.uniform(-0.5, 0.5, int(duration_s * sample_rate)) * 32767).astype("<i2").tobytes()


class FakeBackend:
    """Generic TTSBackend test double — returns a short sine clip as PCM."""
    name = "fake"

    def synthesize(self, text, *, voice=None, language=None, ssml=None):
        from wordbank.backends.base import SynthResult
        sr = 24_000
        t = np.linspace(0, 0.3, int(sr * 0.3), endpoint=False)
        pcm = (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
        return SynthResult(pcm=pcm, sample_rate=sr, billed_chars=len(text))


@pytest.fixture
def silent_wav() -> bytes:
    return make_wav_bytes()


@pytest.fixture
def bank_env(tmp_path, monkeypatch):
    """Point the server at an empty tmp bank (BANK_DB / OGG_ROOT / WAV_ROOT /
    VOICES_ROOT), so no test reads or writes the developer's real bank."""
    paths = types.SimpleNamespace(
        db=tmp_path / "wb.db", ogg=tmp_path / "words_ogg",
        wav=tmp_path / "words", voices=tmp_path / "voices",
    )
    monkeypatch.setenv("BANK_DB", str(paths.db))
    monkeypatch.setenv("OGG_ROOT", str(paths.ogg))
    monkeypatch.setenv("WAV_ROOT", str(paths.wav))
    monkeypatch.setenv("VOICES_ROOT", str(paths.voices))
    for var in ("JWT_SECRET", "GEMINI_API_KEY", "GOOGLE_API_KEY", "MODEL_ENABLED", "SPEAK_STORE"):
        monkeypatch.delenv(var, raising=False)
    return paths


@pytest.fixture
def fake_model(monkeypatch):
    """Stand-in for `src.model`, installed in sys.modules so the server's lazy
    `from .model import ...` binds to it — the suite needs no torch/omnivoice.

    generate() returns PCM that is a pure function of every argument (and of the
    reference clip's bytes), so tests can check what reached the model through
    the audio itself; `calls` also records each call's arguments.
    """
    mod = types.ModuleType("src.model")
    mod.calls = []

    def generate(text, ref_audio_path=None, ref_text=None, instruct=None,
                 guidance_scale=None, seed=None):
        ref_bytes = open(ref_audio_path, "rb").read() if ref_audio_path else None
        mod.calls.append(dict(text=text, ref_audio_path=ref_audio_path, ref_bytes=ref_bytes,
                              ref_text=ref_text, instruct=instruct,
                              guidance_scale=guidance_scale, seed=seed))
        key = f"{text}|{ref_bytes!r}|{ref_text}|{instruct}|{guidance_scale}|{seed}".encode()
        return np.random.RandomState(zlib.crc32(key)).uniform(-1, 1, 4800).astype(np.float32)

    mod.generate = generate
    mod.ensure_loaded = lambda *a, **kw: None
    mod.load_model = lambda *a, **kw: None
    mod.get_device = lambda: "cpu"
    mod.is_loaded = lambda: True
    monkeypatch.setitem(sys.modules, "src.model", mod)
    return mod


@pytest.fixture
def fake_gemini(monkeypatch):
    """Patch Gemini synthesis (the handlers import it lazily from
    wordbank.gemini_tts, so patching the module is what takes effect).
    `calls` records (text, voice, instruct); set `error` to make it raise."""
    import wordbank.gemini_tts as gemini_tts_mod

    state = types.SimpleNamespace(calls=[], error=None)

    def synthesize_pcm(text, voice="Kore", instruct="", model=None):
        state.calls.append((text, voice, instruct))
        if state.error is not None:
            raise state.error
        return noise_pcm16(0.5, seed=zlib.crc32(f"{text}|{voice}|{instruct}".encode()))

    monkeypatch.setattr(gemini_tts_mod, "synthesize_pcm", synthesize_pcm)
    return state


@pytest.fixture
def client(bank_env, fake_model, fake_gemini, monkeypatch):
    """App client over an empty tmp bank, with the model enabled but faked and
    Gemini faked. `client.model` / `client.gemini` expose the fakes.

    Function-scoped ON PURPOSE: `src.server.app` is a module-level singleton and
    TestClient re-runs its lifespan, so a longer-lived client would be torn down
    by any test that opens its own.
    """
    monkeypatch.setenv("MODEL_ENABLED", "1")
    from src.server import app
    with TestClient(app) as c:
        c.model, c.gemini, c.paths = fake_model, fake_gemini, bank_env
        yield c
