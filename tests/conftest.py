import io
import wave

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


@pytest.fixture
def silent_wav() -> bytes:
    return make_wav_bytes()


@pytest.fixture(scope="module")
def client():
    """FastAPI test client with the model enabled but mocked out.

    The synth handler imports src.model lazily, so we patch the names on
    src.model (the lazy `from .model import ...` binds to them at call time).
    MODEL_ENABLED is forced on so /v1/tts is reachable.
    """
    import os
    import src.model as model_mod
    import src.server as server_mod
    from unittest.mock import patch

    def fake_generate(text, ref_audio_path=None, ref_text=None,
                      instruct=None, guidance_scale=None, seed=None):
        return np.zeros(12_000, dtype=np.float32)

    prev = os.environ.get("MODEL_ENABLED")
    os.environ["MODEL_ENABLED"] = "1"
    try:
        with patch.object(model_mod, "_model", object()), \
             patch.object(model_mod, "_device", "cpu"), \
             patch.object(model_mod, "ensure_loaded", lambda *a, **kw: None), \
             patch.object(model_mod, "generate", fake_generate):
            with TestClient(server_mod.app) as c:
                yield c
    finally:
        if prev is None:
            os.environ.pop("MODEL_ENABLED", None)
        else:
            os.environ["MODEL_ENABLED"] = prev
