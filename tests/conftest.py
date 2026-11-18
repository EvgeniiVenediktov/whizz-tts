import base64
import copy
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
    for var in ("JWT_SECRET", "GEMINI_API_KEY", "GOOGLE_API_KEY", "SPEAK_STORE", "GEMINI_TTS_STYLE"):
        monkeypatch.delenv(var, raising=False)
    return paths


@pytest.fixture
def fake_gemini(monkeypatch):
    """Patch Gemini synthesis (the handlers import it lazily from
    wordbank.gemini_tts, so patching the module is what takes effect).
    `calls` records (text, voice, instruct), `kwargs` every call's model and
    style too; set `error` to make it raise."""
    import wordbank.gemini_tts as gemini_tts_mod

    state = types.SimpleNamespace(calls=[], kwargs=[], error=None)

    def synthesize_pcm(text, voice="Kore", instruct="", model=None, style=None,
                       on_request=None):
        state.calls.append((text, voice, instruct))
        state.kwargs.append(dict(text=text, voice=voice, instruct=instruct, model=model,
                                 style=style))
        if state.error is not None:
            raise state.error
        return noise_pcm16(0.5, seed=zlib.crc32(f"{text}|{voice}|{instruct}|{style}".encode()))

    monkeypatch.setattr(gemini_tts_mod, "synthesize_pcm", synthesize_pcm)
    return state


def interaction_resp(audio=None, mime_type="audio/wav", status="completed", errors=None):
    """An Interactions API response as the endpoint sends it: the echoed
    user_input step, then a model_output step holding base64 `audio` (none
    when audio is None)."""
    content = [] if audio is None else [
        {"type": "audio", "data": base64.b64encode(audio).decode(), "mime_type": mime_type}]
    resp = {"id": "v1_test", "status": status,
            "steps": [{"type": "user_input", "content": [{"type": "text", "text": "..."}]},
                      {"type": "model_output", "content": content}],
            "usage": {"total_input_tokens": 3, "total_output_tokens": 40}}
    if errors:
        resp["errors"] = errors
    return resp


@pytest.fixture
def gemini_api(monkeypatch):
    """The REAL gemini_tts.synthesize_pcm over faked transports, so tests see
    what each request path would send. `bodies` records every Interactions
    request body (the style path), `prompts` every generateContent call as
    (model, contents). Interactions answers come from `interactions` (the
    last repeats; an Exception in it is raised), by default 0.5 s of noise as
    a WAV; generateContent answers bare PCM. No network, no retry backoff."""
    import wordbank.gemini_tts as gemini_tts_mod
    from wordbank.gemini_tts import pcm_to_wav_bytes

    pcm = noise_pcm16(0.5, seed=7)
    state = types.SimpleNamespace(pcm=pcm, bodies=[], prompts=[],
                                  interactions=[interaction_resp(pcm_to_wav_bytes(pcm))])

    def post_interaction(body):
        state.bodies.append(copy.deepcopy(body))
        resp = state.interactions[min(len(state.bodies), len(state.interactions)) - 1]
        if isinstance(resp, Exception):
            raise resp
        return resp

    def generate_content(model, contents, config):
        state.prompts.append((model, contents))
        blob = types.SimpleNamespace(data=pcm, mime_type="audio/L16;codec=pcm;rate=24000")
        cand = types.SimpleNamespace(
            content=types.SimpleNamespace(parts=[types.SimpleNamespace(inline_data=blob)]),
            finish_reason="STOP", finish_message=None)
        return types.SimpleNamespace(candidates=[cand], prompt_feedback=None)

    client = types.SimpleNamespace(models=types.SimpleNamespace(generate_content=generate_content))
    monkeypatch.setattr(gemini_tts_mod, "_post_interaction", post_interaction)
    monkeypatch.setattr(gemini_tts_mod, "_get_client", lambda: client)
    monkeypatch.setattr(gemini_tts_mod, "_RETRY_BACKOFF_S", 0)
    monkeypatch.delenv("GEMINI_TTS_EMPTY_RETRIES", raising=False)
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
