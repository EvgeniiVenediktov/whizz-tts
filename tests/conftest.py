import types
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


def noise_pcm16(duration_s: float = 1.0, sample_rate: int = 24_000, seed: int = 0) -> bytes:
    """White noise as 16-bit mono PCM bytes (Gemini's output format). Noise, not
    a tone, so Opus actually spends its bitrate and file size reflects it."""
    rng = np.random.RandomState(seed)
    return (rng.uniform(-0.5, 0.5, int(duration_s * sample_rate)) * 32767).astype("<i2").tobytes()


@pytest.fixture
def bank_env(tmp_path, monkeypatch):
    """Point the server at an empty tmp bank (BANK_DB / OGG_ROOT / WAV_ROOT),
    so no test reads or writes the developer's real bank."""
    paths = types.SimpleNamespace(
        db=tmp_path / "wb.db", ogg=tmp_path / "words_ogg", wav=tmp_path / "words",
    )
    monkeypatch.setenv("BANK_DB", str(paths.db))
    monkeypatch.setenv("OGG_ROOT", str(paths.ogg))
    monkeypatch.setenv("WAV_ROOT", str(paths.wav))
    for var in ("JWT_SECRET", "GEMINI_API_KEY", "GOOGLE_API_KEY", "SPEAK_STORE"):
        monkeypatch.delenv(var, raising=False)
    return paths


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
def client(bank_env, fake_gemini, monkeypatch):
    """App client over an empty tmp bank with Gemini faked (`client.gemini`),
    configured like a Gemini deployment (a key is set, so /health is green).

    Function-scoped ON PURPOSE: `src.server.app` is a module-level singleton and
    TestClient re-runs its lifespan, so a longer-lived client would be torn down
    by any test that opens its own.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    from src.server import app
    with TestClient(app) as c:
        c.gemini, c.paths = fake_gemini, bank_env
        yield c
