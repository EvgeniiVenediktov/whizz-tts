"""Voice studio: the Gemini voice list and /v1/gemini/tts previews (Gemini faked)."""
from wordbank.gemini_tts import canonical_voice


def test_every_listed_voice_is_accepted_by_name(client):
    voices = client.get("/v1/gemini/voices").json()["voices"]
    assert len(voices) == 30 and all(set(v) == {"name", "style", "gender"} for v in voices)
    assert all(v["gender"] in ("male", "female") and v["style"] for v in voices)
    for v in voices:   # the list the UI offers is exactly what the API validates
        assert canonical_voice(v["name"]) == v["name"]
        assert canonical_voice(v["name"].upper()) == v["name"]


def test_gemini_tts_returns_wav_in_the_requested_voice_and_saves_nothing(client):
    r = client.post("/v1/gemini/tts", data={"text": "a unique experience", "voice": "puck",
                                            "instruct": "warmly"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav") and r.content[:4] == b"RIFF"
    assert client.gemini.calls == [("a unique experience", "Puck", "warmly")]
    assert not client.paths.db.exists()


def test_gemini_tts_unknown_voice_is_400(client):
    r = client.post("/v1/gemini/tts", data={"text": "hi", "voice": "Bogus"})
    assert r.status_code == 400
    assert client.gemini.calls == []


def test_gemini_tts_upstream_failure_is_502(client):
    client.gemini.error = RuntimeError("No API key was provided")
    r = client.post("/v1/gemini/tts", data={"text": "hi"})
    assert r.status_code == 502 and "No API key" in r.json()["detail"]
