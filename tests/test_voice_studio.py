"""Voice studio: the Gemini voice list, /v1/gemini/tts, saved clone references,
and /v1/tts cloning from a saved reference (Gemini and the model faked)."""
import pytest

from tests.conftest import make_wav_bytes
from wordbank.gemini_tts import canonical_voice
from wordbank.manifest import Manifest


def test_every_listed_voice_is_accepted_by_name(client):
    voices = client.get("/v1/gemini/voices").json()["voices"]
    assert len(voices) == 30
    assert all(v["gender"] in ("male", "female") and v["style"] for v in voices)
    for v in voices:   # the list the UI offers is exactly what the API validates
        assert canonical_voice(v["name"]) == v["name"]
        assert canonical_voice(v["name"].upper()) == v["name"]


def test_gemini_tts_returns_wav_in_the_requested_voice(client):
    r = client.post("/v1/gemini/tts", data={"text": "a unique experience", "voice": "puck",
                                            "instruct": "warmly"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav") and r.content[:4] == b"RIFF"
    assert client.gemini.calls == [("a unique experience", "Puck", "warmly")]


def test_gemini_tts_unknown_voice_is_400(client):
    r = client.post("/v1/gemini/tts", data={"text": "hi", "voice": "Bogus"})
    assert r.status_code == 400
    assert client.gemini.calls == []


def test_gemini_tts_upstream_failure_is_502(client):
    client.gemini.error = RuntimeError("No API key was provided")
    r = client.post("/v1/gemini/tts", data={"text": "hi"})
    assert r.status_code == 502 and "No API key" in r.json()["detail"]


def test_save_list_and_serve_voice(client):
    wav = make_wav_bytes(0.2)
    r = client.post(
        "/v1/voices",
        data={"name": "kore", "lang": "en-us", "ref_text": "hi there", "source": "gemini"},
        files={"audio": ("kore.wav", wav, "audio/wav")},
    )
    assert r.status_code == 200
    assert r.json()["ref_audio"] == "en-us/kore.wav"
    assert (client.paths.voices / "en-us" / "kore.wav").read_bytes() == wav

    voices = client.get("/v1/voices").json()["voices"]
    assert any(v["name"] == "kore" and v["lang"] == "en-us" for v in voices)
    assert client.get("/v1/voices", params={"lang": "en-gb"}).json()["voices"] == []
    assert client.get("/v1/voices/en-us/kore/audio").content == wav


def test_save_voice_rejects_non_wav(client):
    r = client.post("/v1/voices", data={"name": "x", "lang": "en-us", "ref_text": "y"},
                    files={"audio": ("x.wav", b"NOT_A_WAV", "audio/wav")})
    assert r.status_code == 400
    assert client.get("/v1/voices").json()["voices"] == []


def test_tts_clones_from_saved_voice(client):
    clip = make_wav_bytes(0.2)
    (client.paths.voices / "en-us").mkdir(parents=True)
    (client.paths.voices / "en-us" / "kore.wav").write_bytes(clip)
    m = Manifest(str(client.paths.db))
    m.add_voice("kore", "en-us", "en-us/kore.wav", ref_text="hi there")
    m.close()

    r = client.post("/v1/tts", data={"text": "hello", "voice": "kore", "lang": "en-us"})
    assert r.status_code == 200, r.text
    [call] = client.model.calls
    assert call["ref_bytes"] == clip and call["ref_text"] == "hi there"


@pytest.mark.parametrize("on_disk", [False, True])
def test_tts_saved_voice_missing_is_404(client, on_disk):
    if on_disk:   # registered, but the clip file is gone
        m = Manifest(str(client.paths.db))
        m.add_voice("kore", "en-us", "en-us/kore.wav")
        m.close()
    r = client.post("/v1/tts", data={"text": "hello", "voice": "kore", "lang": "en-us"})
    assert r.status_code == 404
    assert client.model.calls == []
