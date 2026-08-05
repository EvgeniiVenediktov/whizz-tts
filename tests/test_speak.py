"""Tests for the get-or-generate /v1/speak endpoint: bank hit vs. Gemini miss,
persistence into the phrase bank, and auth/validation."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from wordbank.manifest import DONE, Manifest


def _fake_pcm(duration_s: float = 0.2, sample_rate: int = 24_000) -> bytes:
    """Short 24 kHz sine as 16-bit mono PCM bytes, standing in for Gemini output."""
    n = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n, endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16).tobytes()


@pytest.fixture
def speak_client(tmp_path, monkeypatch):
    """TestClient wired to an empty tmp bank + WAV/OGG roots, with Gemini
    synthesis mocked. The handler's lazy `from wordbank.gemini_tts import
    synthesize_pcm` binds at call time, so patching the module (not
    src.server) is what actually takes effect."""
    import wordbank.gemini_tts as gemini_tts_mod

    calls = []

    def fake_synthesize_pcm(text, voice="Kore", instruct="", model=None):
        calls.append((text, voice, instruct))
        return _fake_pcm()

    monkeypatch.setattr(gemini_tts_mod, "synthesize_pcm", fake_synthesize_pcm)

    wav_root = tmp_path / "words"
    wav_root.mkdir()   # dev WAV root present -> also exercises the WAV-master write
    monkeypatch.setenv("BANK_DB", str(tmp_path / "wb.db"))
    monkeypatch.setenv("OGG_ROOT", str(tmp_path / "words_ogg"))
    monkeypatch.setenv("WAV_ROOT", str(wav_root))

    from src.server import app
    with TestClient(app) as c:
        c.calls = calls
        yield c


def test_speak_miss_generates_and_persists(speak_client, tmp_path):
    from src.server import GEMINI_TTS_VOICE

    r = speak_client.post("/v1/speak", data={"text": "hello there", "lang": "en-us"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/ogg")
    assert r.headers["x-source"] == "gemini"
    assert r.content and r.content[:4] == b"OggS"
    assert len(speak_client.calls) == 1

    m = Manifest(str(tmp_path / "wb.db"), table="phrases", text_col="text")
    row = m.get_entry("hello there", "en-us", f"gemini-{GEMINI_TTS_VOICE}")
    assert row is not None
    assert row["status"] == DONE
    assert row["voice"] == f"gemini-{GEMINI_TTS_VOICE}"

    rel_wav = row["file_path"]
    ogg_path = tmp_path / "words_ogg" / (rel_wav[:-4] + ".ogg")
    assert ogg_path.exists()
    wav_path = tmp_path / "words" / rel_wav
    assert wav_path.exists()   # best-effort dev master, WAV_ROOT was present
    m.close()


def test_speak_second_request_serves_from_bank(speak_client):
    r1 = speak_client.post("/v1/speak", data={"text": "hello there", "lang": "en-us"})
    assert r1.headers["x-source"] == "gemini"

    r2 = speak_client.post("/v1/speak", data={"text": "hello there", "lang": "en-us"})
    assert r2.status_code == 200
    assert r2.headers["x-source"] == "bank"
    assert r2.content[:4] == b"OggS"
    assert len(speak_client.calls) == 1   # Gemini was NOT called again


def test_speak_serves_existing_word_from_bank(speak_client, tmp_path):
    ogg = tmp_path / "words_ogg" / "en-us/gemini-Kore/0/1_cat.ogg"
    ogg.parent.mkdir(parents=True)
    ogg.write_bytes(b"OggS" + b"\x00" * 50)

    m = Manifest(str(tmp_path / "wb.db"))
    m.add_words(["cat"], "en-us")
    row_id = m.iter_rows()[0]["id"]
    m.set_status(row_id, DONE, file_path="en-us/gemini-Kore/0/1_cat.wav", voice="Kore")
    m.close()

    r = speak_client.post("/v1/speak", data={"text": "cat", "lang": "en-us"})
    assert r.status_code == 200
    assert r.headers["x-source"] == "bank"
    assert r.content[:4] == b"OggS"
    assert speak_client.calls == []   # no Gemini call for a word-bank hit


def test_speak_generated_phrase_appears_in_search(speak_client):
    speak_client.post("/v1/speak", data={"text": "hello there", "lang": "en-us"})
    results = speak_client.get(
        "/v1/search", params={"prefix": "hello", "lang": "en-us"}
    ).json()["results"]
    assert any(r["text"] == "hello there" and r["type"] == "phrase" for r in results)


def test_speak_requires_auth_when_secret_set(speak_client, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret-0123456789abcdef0123456789abcdef")
    r = speak_client.post("/v1/speak", data={"text": "hello there"})
    assert r.status_code == 401
    assert speak_client.calls == []


def test_speak_text_too_long_400(speak_client):
    r = speak_client.post("/v1/speak", data={"text": "a" * 2001})
    assert r.status_code == 400
    assert speak_client.calls == []


def test_speak_empty_text_400(speak_client):
    r = speak_client.post("/v1/speak", data={"text": "   "})
    assert r.status_code == 400
    assert speak_client.calls == []
