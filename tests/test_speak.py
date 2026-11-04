"""/v1/speak get-or-generate: voice selection, bank hit vs Gemini miss,
persistence, and the saved format. Real handler over a tmp bank; Gemini faked."""
import io

import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from wordbank.manifest import DONE, Manifest


def _speak(client, **data):
    return client.post("/v1/speak", data=data)


def _phrase_rows(paths):
    m = Manifest(str(paths.db), table="phrases", text_col="text")
    try:
        return [dict(r) for r in m.iter_rows()]
    finally:
        m.close()


def _bank_word(paths, word, voice, file_voice="Kore"):
    """A done word with its OGG on disk; `voice=None` mimics a bank predating
    the voice column (only the file path says which voice made it)."""
    rel = f"en-us/gemini-{file_voice}/0/1_{word}.wav"
    ogg = paths.ogg / (rel[:-4] + ".ogg")
    ogg.parent.mkdir(parents=True, exist_ok=True)
    ogg.write_bytes(b"OggS-word-" + word.encode())
    m = Manifest(str(paths.db))
    m.add_words([word], "en-us", voice=voice)
    m.set_status(m.iter_rows()[0]["id"], DONE, file_path=rel, voice=voice)
    m.close()
    return ogg.read_bytes()


def _bank_phrase(paths, text, voice):
    rel = f"en-us/{voice}/0/{voice}.wav"
    ogg = paths.ogg / (rel[:-4] + ".ogg")
    ogg.parent.mkdir(parents=True, exist_ok=True)
    ogg.write_bytes(b"OggS-" + voice.encode())
    m = Manifest(str(paths.db), table="phrases", text_col="text")
    m.add_words([text], "en-us", voice=voice)
    m.set_status(m.get_entry(text, "en-us", voice)["id"], DONE, file_path=rel, voice=voice)
    m.close()
    return ogg.read_bytes()


def test_miss_generates_in_selected_voice_and_persists(client):
    r = _speak(client, text="hello there", voice="Puck")

    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("audio/ogg")
    assert r.headers["x-source"] == "gemini" and r.headers["x-voice"] == "gemini-Puck"
    assert client.gemini.calls == [("hello there", "Puck", "")]
    pcm, sr = sf.read(io.BytesIO(r.content))
    assert abs(len(pcm) / sr - 0.5) < 0.05          # the whole 0.5 s clip decodes

    [row] = _phrase_rows(client.paths)
    assert row["status"] == DONE and row["voice"] == "gemini-Puck"
    assert row["file_path"].startswith("en-us/gemini-Puck/phrases/")
    saved = client.paths.ogg / (row["file_path"][:-4] + ".ogg")
    assert saved.read_bytes() == r.content          # what was served is what was banked
    assert not (client.paths.wav / row["file_path"]).exists()   # no WAV root -> no master


def test_repeat_is_a_bank_hit_and_each_voice_is_banked_separately(client):
    first = _speak(client, text="hello there", voice="Puck")
    again = _speak(client, text="Hello there", voice="puck")   # text + voice case-insensitive
    assert again.headers["x-source"] == "bank" and again.content == first.content

    other = _speak(client, text="hello there", voice="Charon")
    assert other.headers["x-source"] == "gemini" and other.headers["x-voice"] == "gemini-Charon"

    assert [voice for _, voice, _ in client.gemini.calls] == ["Puck", "Charon"]
    assert sorted(r["voice"] for r in _phrase_rows(client.paths)) == ["gemini-Charon", "gemini-Puck"]
    results = client.get("/v1/search", params={"prefix": "hello"}).json()["results"]
    assert {r["voice"] for r in results} == {"gemini-Puck", "gemini-Charon"}


def test_unknown_voice_is_rejected_before_calling_gemini(client):
    r = _speak(client, text="hello there", voice="Bogus")
    assert r.status_code == 400 and "Bogus" in r.json()["detail"]
    assert client.gemini.calls == []


def test_word_bank_hit_honours_the_selected_voice(client):
    word_ogg = _bank_word(client.paths, "cat", voice="Kore")

    for data in ({"voice": "kore"}, {}):                        # same voice, or none
        r = _speak(client, text="cat", **data)
        assert r.headers["x-source"] == "bank" and r.content == word_ogg
        assert r.headers["x-voice"] == "Kore"
    assert client.gemini.calls == []

    r = _speak(client, text="cat", voice="Puck")                # banked word is Kore
    assert r.headers["x-source"] == "gemini" and r.headers["x-voice"] == "gemini-Puck"
    assert client.gemini.calls == [("cat", "Puck", "")]


def test_word_without_recorded_voice_matches_on_its_file_path(client):
    word_ogg = _bank_word(client.paths, "cat", voice=None, file_voice="Kore")

    assert _speak(client, text="cat", voice="Kore").content == word_ogg
    assert _speak(client, text="cat", voice="Puck").headers["x-source"] == "gemini"


def test_no_voice_prefers_the_default_voice_then_any_banked(client):
    _bank_phrase(client.paths, "hello there", "gemini-Puck")    # banked first
    kore = _bank_phrase(client.paths, "hello there", "gemini-Kore")
    r = _speak(client, text="hello there")
    assert r.content == kore and r.headers["x-voice"] == "gemini-Kore"

    local = _bank_phrase(client.paths, "good morning", "local-en-gb")
    r = _speak(client, text="good morning")
    assert r.content == local and r.headers["x-voice"] == "local-en-gb"
    assert client.gemini.calls == []

    r = _speak(client, text="good morning", voice="Kore")       # pinned: local clip doesn't count
    assert r.headers["x-source"] == "gemini" and r.headers["x-voice"] == "gemini-Kore"


def test_instruct_is_a_one_off_render_never_banked(client):
    _bank_phrase(client.paths, "hello there", "gemini-Puck")

    for _ in range(2):
        r = _speak(client, text="hello there", voice="Puck", instruct="whisper it")
        assert r.status_code == 200 and r.headers["x-source"] == "gemini"
    assert client.gemini.calls == [("hello there", "Puck", "whisper it")] * 2
    assert [r["voice"] for r in _phrase_rows(client.paths)] == ["gemini-Puck"]  # unchanged


def test_banked_row_whose_ogg_is_missing_is_regenerated(client):
    _speak(client, text="hello there", voice="Puck")
    [row] = _phrase_rows(client.paths)
    ogg = client.paths.ogg / (row["file_path"][:-4] + ".ogg")
    ogg.unlink()

    r = _speak(client, text="hello there", voice="Puck")
    assert r.headers["x-source"] == "gemini" and ogg.read_bytes() == r.content
    assert len(_phrase_rows(client.paths)) == 1


def test_gemini_failure_is_502_and_saves_nothing(client):
    client.gemini.error = RuntimeError("quota exhausted")
    r = _speak(client, text="hello there", voice="Puck")
    assert r.status_code == 502 and "quota exhausted" in r.json()["detail"]
    assert _phrase_rows(client.paths) == []


def test_saved_clip_uses_the_bank_ogg_profile(client):
    """Banked clips match the batch bank (16k libopus), not the 64k live default."""
    r = _speak(client, text="hello there", voice="Puck")        # 0.5 s of noise
    kbps = len(r.content) * 8 / 0.5 / 1000
    assert kbps < 24, f"{kbps:.0f} kbps — not the 16k bank profile"


def test_missing_ffmpeg_is_503_and_saves_nothing(client, monkeypatch):
    monkeypatch.setenv("PATH", "")
    r = _speak(client, text="hello there", voice="Puck")
    assert r.status_code == 503 and "ffmpeg" in r.json()["detail"]
    assert _phrase_rows(client.paths) == []


@pytest.mark.parametrize("text", ["   ", "a" * 2001])
def test_empty_or_too_long_text_is_400(client, text):
    assert _speak(client, text=text).status_code == 400
    assert client.gemini.calls == []


def test_bankless_deploy_first_write_creates_the_bank(bank_env, fake_gemini, monkeypatch, tmp_path):
    """No bank yet, only a Gemini key: healthy, and the first miss creates the DB
    (parent dir included) so later requests are bank hits."""
    db = tmp_path / "fresh" / "volume" / "wordbank.db"
    monkeypatch.setenv("BANK_DB", str(db))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    from src.server import app
    with TestClient(app) as c:
        health = c.get("/health")
        assert health.status_code == 200 and health.json()["bank"] is False

        assert _speak(c, text="hello there").headers["x-source"] == "gemini"
        assert db.exists()
        assert _speak(c, text="hello there").headers["x-source"] == "bank"
        assert c.get("/health").json()["bank"] is True


# --- SPEAK_STORE=0: stateless upstream (generates, never writes) -------------

def _ogg_files(paths):
    return list(paths.ogg.rglob("*.ogg")) if paths.ogg.exists() else []


def test_store_off_serves_the_miss_and_writes_nothing(client, monkeypatch):
    monkeypatch.setenv("SPEAK_STORE", "0")
    r = _speak(client, text="hello there", voice="Puck")
    assert r.status_code == 200, r.text
    assert r.headers["x-source"] == "gemini" and r.headers["x-voice"] == "gemini-Puck"
    assert r.content[:4] == b"OggS"
    assert _phrase_rows(client.paths) == [] and _ogg_files(client.paths) == []

    again = _speak(client, text="hello there", voice="Puck")   # nothing banked -> generated again
    assert again.status_code == 200 and again.headers["x-source"] == "gemini"
    assert len(client.gemini.calls) == 2
    assert client.get("/health").json()["store"] is False


def test_store_off_still_serves_an_existing_bank(client, monkeypatch):
    banked = _bank_phrase(client.paths, "hello there", "gemini-Puck")
    monkeypatch.setenv("SPEAK_STORE", "0")
    r = _speak(client, text="hello there", voice="Puck")
    assert r.headers["x-source"] == "bank" and r.content == banked
    assert client.gemini.calls == []


def test_store_off_gemini_failure_is_502(client, monkeypatch):
    monkeypatch.setenv("SPEAK_STORE", "0")
    client.gemini.error = RuntimeError("quota exhausted")
    r = _speak(client, text="hello there", voice="Puck")
    assert r.status_code == 502 and "quota exhausted" in r.json()["detail"]
    client.gemini.error = None
    assert _speak(client, text="hello there", voice="Puck").status_code == 200   # failure not cached


def test_store_off_concurrent_misses_share_one_generation(client, monkeypatch):
    """No saved clip for queued twins to find, so the burst shares one result."""
    import time
    from concurrent.futures import ThreadPoolExecutor
    import wordbank.gemini_tts as gemini_tts_mod

    monkeypatch.setenv("SPEAK_STORE", "0")
    fast = gemini_tts_mod.synthesize_pcm

    def slow(*a, **kw):
        time.sleep(0.3)
        return fast(*a, **kw)
    monkeypatch.setattr(gemini_tts_mod, "synthesize_pcm", slow)

    with ThreadPoolExecutor(max_workers=6) as pool:
        rs = list(pool.map(lambda _: _speak(client, text="hello there", voice="Puck"), range(6)))
    assert all(r.status_code == 200 for r in rs)
    assert len({r.content for r in rs}) == 1
    assert len(client.gemini.calls) == 1
    assert _phrase_rows(client.paths) == []
