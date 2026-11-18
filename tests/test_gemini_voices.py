"""Curated per-language Gemini voices: the data (wordbank/gemini_voices.json),
their labels, the 3.8-only rule for locale voices, the routes that expose and
accept them, and scripts/fetch_gemini_voices.py. Gemini is faked throughout.

Why the labels matter: banks already key on the classic capitalized names
(`words.voice = "Kore"`, phrases `gemini-Kore`, paths `en/gemini-Kore/…`), while
the batch path files a locale voice under its lowercase id
(`en-gb/gemini-en-gb-tutor-11/…`); /v1/speak must use the same labels.
"""
import importlib.util
import io
import json
import pathlib

import pytest

import src.server as srv
from wordbank.gemini_tts import GeminiVoiceUnsupportedError, canonical_voice, synthesize_pcm
from wordbank.languages import VOICE_LANGUAGES, is_lang_code
from wordbank.manifest import Manifest

LOCALE = "en-gb-tutor-11"
PREVIEW = "gemini-3.1-flash-tts-preview"
SCRIPT = pathlib.Path(__file__).parent.parent / "scripts" / "fetch_gemini_voices.py"


def test_curated_voices_are_one_female_and_one_male_per_language():
    assert [lang["code"] for lang in VOICE_LANGUAGES] == [
        "en-us", "en-gb", "ru-ru", "fr-fr", "de-de", "zh-cn", "ja-jp", "es-es"]
    for lang in VOICE_LANGUAGES:
        assert is_lang_code(lang["code"]) and lang["name"]
        assert [v["gender"] for v in lang["voices"]] == ["female", "male"], lang["code"]
        assert all(v["display_name"] and v["accent"] and v["description"] for v in lang["voices"])
    # en-us is two classic voices, labelled as banks already key them
    assert [canonical_voice(v["id"]) for v in VOICE_LANGUAGES[0]["voices"]] == ["Kore", "Iapetus"]


def test_canonical_voice_classic_vs_locale():
    assert canonical_voice(" pUCK ") == "Puck"
    assert canonical_voice("EN-GB-Tutor-11") == LOCALE
    assert canonical_voice("en-gb-advisor-1") is None      # in Gemini's library, not curated
    assert canonical_voice("Bogus") is None


def test_locale_voice_on_a_non_38_model_is_refused_before_any_request(gemini_api):
    with pytest.raises(GeminiVoiceUnsupportedError, match="verified only on"):
        synthesize_pcm("hello", voice=LOCALE, model=PREVIEW)
    assert gemini_api.bodies == [] and gemini_api.prompts == []


def test_routes_refuse_a_locale_voice_on_a_non_38_model(client, monkeypatch):
    studio = client.post("/v1/gemini/tts", data={"text": "hi", "voice": LOCALE, "model": PREVIEW})
    monkeypatch.setattr(srv, "GEMINI_TTS_MODEL", PREVIEW)
    speak = client.post("/v1/speak", data={"text": "hello there", "voice": LOCALE})
    classic = client.post("/v1/speak", data={"text": "hello there", "voice": "Kore"})
    assert studio.status_code == 400 and "verified only on" in studio.json()["detail"]
    assert speak.status_code == 400 and "verified only on" in speak.json()["detail"]
    assert classic.status_code == 200
    assert client.gemini.calls == [("hello there", "Kore", "")]


def test_cli_refuses_a_locale_voice_on_a_non_38_model(tmp_path):
    from wordbank.__main__ import main
    words = tmp_path / "w.txt"
    words.write_text("hello\n")
    db = tmp_path / "wb.db"
    with pytest.raises(SystemExit, match="verified only on"):
        main(["--db", str(db), "--out", str(tmp_path / "out"), "--lang", "en-gb",
              "batch-submit", "--words", str(words), "--voice", LOCALE])
    assert not db.exists()                                  # refused before ingesting


def test_speak_files_a_locale_voice_under_its_id(client):
    miss = client.post("/v1/speak", data={"text": "hello there", "lang": "en-gb",
                                          "voice": "EN-GB-Tutor-11"})
    hit = client.post("/v1/speak", data={"text": "hello there", "lang": "en-gb", "voice": LOCALE})
    assert miss.status_code == 200, miss.text
    assert (miss.headers["x-source"], miss.headers["x-voice"]) == ("gemini", f"gemini-{LOCALE}")
    assert hit.headers["x-source"] == "bank" and hit.content == miss.content
    assert client.gemini.calls == [("hello there", LOCALE, "")]
    m = Manifest(str(client.paths.db), table="phrases", text_col="text")
    [row] = m.iter_rows()
    m.close()
    assert row["voice"] == f"gemini-{LOCALE}"
    assert row["file_path"].startswith(f"en-gb/gemini-{LOCALE}/phrases/")


def test_gemini_languages_lists_the_curated_voices(client):
    body = client.get("/v1/gemini/languages").json()
    langs = {lang["code"]: lang for lang in body["languages"]}
    assert len(langs) == 8 and "gemini-3.1-flash-tts-preview" not in body["locale_voice_models"]
    female, male = langs["en-gb"]["voices"]
    assert (female["name"], female["gender"], female["classic"], female["accent"]) == \
        (LOCALE, "female", False, "Winchester English")
    assert male["gender"] == "male" and male["display_name"] and male["description"]
    assert [(v["name"], v["classic"]) for v in langs["en-us"]["voices"]] == \
        [("Kore", True), ("Iapetus", True)]


# --- scripts/fetch_gemini_voices.py ------------------------------------------------

def _script():
    spec = importlib.util.spec_from_file_location("fetch_gemini_voices", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _voice(vid, gender, lang="en-GB"):
    return {"id": vid, "type": "prebuilt", "display_name": vid.title(), "language_code": lang,
            "region_code": lang[-2:], "accent": "Winchester English", "persona": "Tutor",
            "context": "Edu", "gender": gender, "pitch": "low", "description": f"{vid} voice"}


def test_fetch_script_refreshes_the_picks_from_a_saved_library(tmp_path):
    library = tmp_path / "raw.json"
    library.write_text(json.dumps({"voices": [
        _voice("en-gb-a-1", "male"), _voice("en-gb-b-2", "female"), _voice("en-gb-c-3", "male")]}))
    curated = tmp_path / "curated.json"
    curated.write_text(json.dumps({"languages": [
        {"code": "en-gb", "name": "English (UK)", "voices": [{"id": "en-gb-a-1"}, {"id": "en-gb-b-2"}]}]}))

    _script().main(["--from-file", str(library), "--out", str(curated), "--fetched", "2026-01-02"])
    data = json.loads(curated.read_text())
    assert data["fetched"] == "2026-01-02" and data["source"].endswith("/v1beta/voices")
    [lang] = data["languages"]
    assert [v["id"] for v in lang["voices"]] == ["en-gb-b-2", "en-gb-a-1"]     # female first
    assert lang["voices"][0] == {"id": "en-gb-b-2", "gender": "female", "display_name": "En-Gb-B-2",
                                 "language_code": "en-GB", "accent": "Winchester English",
                                 "pitch": "low", "persona": "Tutor", "description": "en-gb-b-2 voice"}

    lang["voices"][0]["id"] = "en-gb-c-3"                      # two male picks: refused
    curated.write_text(json.dumps(data))
    with pytest.raises(SystemExit, match="one female and one male"):
        _script().main(["--from-file", str(library), "--out", str(curated)])
    assert json.loads(curated.read_text()) == data


def test_fetch_voices_follows_next_page_token():
    pages = [{"voices": [_voice("a-1", "male")], "next_page_token": "p2"},
             {"voices": [_voice("b-2", "female")]}]
    requests = []

    def opener(req, timeout):
        requests.append(req)
        return io.BytesIO(json.dumps(pages[len(requests) - 1]).encode())

    voices = _script().fetch_voices("k", opener=opener)
    assert [v["id"] for v in voices] == ["a-1", "b-2"]
    assert [r.full_url.split("?")[1] for r in requests] == [
        "page_size=1000", "page_size=1000&page_token=p2"]
    assert all(r.full_url.startswith("https://generativelanguage.googleapis.com/v1beta/voices?")
               and r.get_header("X-goog-api-key") == "k" for r in requests)
