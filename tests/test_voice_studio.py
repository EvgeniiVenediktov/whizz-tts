"""Server tests for the Gemini voice studio + saved-reference endpoints
(Gemini and the OmniVoice model are mocked)."""
import numpy as np
from fastapi.testclient import TestClient

from tests.conftest import make_wav_bytes
from wordbank.manifest import Manifest


def test_gemini_voices_list(tmp_path, monkeypatch):
    monkeypatch.setenv("BANK_DB", str(tmp_path / "wb.db"))
    from src.server import app
    with TestClient(app) as c:
        voices = c.get("/v1/gemini/voices").json()["voices"]
        assert len(voices) == 30
        names = {v["name"] for v in voices}
        assert {"Kore", "Puck", "Zephyr"} <= names
        kore = next(v for v in voices if v["name"] == "Kore")
        assert kore["style"] == "Firm" and kore["gender"] == "female"
        # every voice is marked male/female
        assert all(v["gender"] in ("male", "female") for v in voices)
        genders = [v["gender"] for v in voices]
        assert genders.count("female") == 14 and genders.count("male") == 16


def test_gemini_tts_returns_wav(tmp_path, monkeypatch):
    import wordbank.gemini_tts as g
    monkeypatch.setattr(g, "synthesize_pcm", lambda *a, **k: b"\x00\x00" * 2400)
    monkeypatch.setenv("BANK_DB", str(tmp_path / "wb.db"))
    monkeypatch.setenv("VOICES_ROOT", str(tmp_path / "voices"))

    from src.server import app
    with TestClient(app) as c:
        r = c.post("/v1/gemini/tts", data={"text": "this was such a unique experience"})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("audio/wav")
        assert r.content[:4] == b"RIFF"


def test_save_list_and_serve_voice(tmp_path, monkeypatch):
    monkeypatch.setenv("BANK_DB", str(tmp_path / "wb.db"))
    monkeypatch.setenv("VOICES_ROOT", str(tmp_path / "voices"))

    from src.server import app
    wav = make_wav_bytes(0.2)
    with TestClient(app) as c:
        r = c.post(
            "/v1/voices",
            data={"name": "kore", "lang": "en-us", "ref_text": "hi there", "source": "gemini"},
            files={"audio": ("kore.wav", wav, "audio/wav")},
        )
        assert r.status_code == 200
        assert r.json()["ref_audio"] == "en-us/kore.wav"
        assert (tmp_path / "voices" / "en-us" / "kore.wav").exists()

        voices = c.get("/v1/voices").json()["voices"]
        assert any(v["name"] == "kore" and v["lang"] == "en-us" for v in voices)
        assert c.get("/v1/voices", params={"lang": "en-gb"}).json()["voices"] == []

        audio = c.get("/v1/voices/en-us/kore/audio")
        assert audio.status_code == 200 and audio.content[:4] == b"RIFF"

        # non-WAV upload is rejected
        bad = c.post(
            "/v1/voices",
            data={"name": "x", "lang": "en-us", "ref_text": "y"},
            files={"audio": ("x.wav", b"NOT_A_WAV", "audio/wav")},
        )
        assert bad.status_code == 400


def test_tts_uses_saved_voice(tmp_path, monkeypatch):
    import src.model as model_mod
    from unittest.mock import patch

    # Seed a saved voice + its clip on disk.
    (tmp_path / "voices" / "en-us").mkdir(parents=True)
    (tmp_path / "voices" / "en-us" / "kore.wav").write_bytes(make_wav_bytes(0.2))
    m = Manifest(str(tmp_path / "wb.db"))
    m.add_voice("kore", "en-us", "en-us/kore.wav", ref_text="hi there")
    m.close()

    monkeypatch.setenv("BANK_DB", str(tmp_path / "wb.db"))
    monkeypatch.setenv("VOICES_ROOT", str(tmp_path / "voices"))
    monkeypatch.setenv("MODEL_ENABLED", "1")

    seen = {}

    def fake_generate(text, ref_audio_path=None, ref_text=None, instruct=None,
                      guidance_scale=None, seed=None):
        seen["ref_audio_path"] = ref_audio_path
        seen["ref_text"] = ref_text
        return np.zeros(2400, dtype=np.float32)

    from src.server import app
    with patch.object(model_mod, "_model", object()), \
         patch.object(model_mod, "ensure_loaded", lambda *a, **k: None), \
         patch.object(model_mod, "generate", fake_generate):
        with TestClient(app) as c:
            r = c.post("/v1/tts", data={"text": "hello", "voice": "kore", "lang": "en-us"})
            assert r.status_code == 200, r.text
            assert r.headers["content-type"].startswith("audio/ogg")

    # The saved clip's resolved path + stored transcript reached the model.
    assert seen["ref_audio_path"].endswith("en-us/kore.wav")
    assert seen["ref_text"] == "hi there"


def test_tts_unknown_saved_voice_404(tmp_path, monkeypatch):
    import src.model as model_mod
    from unittest.mock import patch

    monkeypatch.setenv("BANK_DB", str(tmp_path / "wb.db"))
    monkeypatch.setenv("VOICES_ROOT", str(tmp_path / "voices"))
    monkeypatch.setenv("MODEL_ENABLED", "1")

    from src.server import app
    with patch.object(model_mod, "_model", object()), \
         patch.object(model_mod, "ensure_loaded", lambda *a, **k: None), \
         patch.object(model_mod, "generate", lambda *a, **k: np.zeros(10, dtype=np.float32)):
        with TestClient(app) as c:
            r = c.post("/v1/tts", data={"text": "hello", "voice": "ghost", "lang": "en-us"})
            assert r.status_code == 404
