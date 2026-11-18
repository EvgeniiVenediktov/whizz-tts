"""Style directions for Gemini TTS: the capability registry and routing, the
Interactions request synthesize_pcm sends, its transport, and how /v1/speak,
GEMINI_TTS_STYLE and the studio use it.

Why it matters: the 3.8 models read inline directions ALOUD ("Say …: laughing"
came back as the whole prompt spoken) and act out bare words ("haha" became a
laugh track); their only safe direction is the speech_metadata style, with the
text kept verbatim. 3.1 rejects that annotation with a 400.

The real synthesize_pcm runs over the faked transports of `gemini_api`.
"""
import io
import json
import logging
import types
import urllib.error

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("google.genai.types")      # the generateContent path builds SDK config

import wordbank.gemini_tts as gemini_tts  # noqa: E402
from tests.conftest import interaction_resp  # noqa: E402
from tests.test_gemini_payload import gemini_lite_wav  # noqa: E402
from wordbank.gemini_tts import (  # noqa: E402
    GeminiAPIError, GeminiNetworkError, GeminiNoAudioError, GeminiStyleUnsupportedError,
    route_direction, synthesize_pcm,
)
from wordbank.manifest import DONE, Manifest  # noqa: E402

PCM = bytes(range(256)) * 40
STYLE = ("read the word exactly as written, plain and neutral, in English; "
         "no acting, laughing or translating")
LITE, FLASH = "gemini-3.8-flash-lite-tts", "gemini-3.8-flash-tts"
PREVIEW = "gemini-3.1-flash-tts-preview"


def _sent(api):
    """What reached Gemini: ('style', text, style) per Interactions request,
    ('generate', contents) per generateContent call."""
    return ([("style", b["input"][0]["content"][0]["text"],
              b["input"][0]["content"][0]["annotations"][0]["style"]) for b in api.bodies]
            + [("generate", contents) for _, contents in api.prompts])


# --- registry and routing --------------------------------------------------------

def test_capability_registry():
    assert gemini_tts.style_models() == sorted([FLASH, LITE])
    assert gemini_tts.model_caps(f"models/{LITE}") == gemini_tts.TTSModel(style=True, inline=False)
    for model in (PREVIEW, "gemini-2.5-pro-preview-tts", "gemini-9-unlisted-tts"):
        assert gemini_tts.model_caps(model) == gemini_tts.TTSModel(style=False, inline=True), model


def test_route_direction():
    """-> (style, inline prefix). 3.8 never gets an inline prefix."""
    assert route_direction(LITE, "", STYLE) == (STYLE, "")
    assert route_direction(LITE, "whisper it") == ("whisper it", "")
    assert route_direction(FLASH, "whisper it", "plain and neutral") == \
        ("plain and neutral. whisper it", "")
    assert route_direction(FLASH) == (None, "")
    assert route_direction(PREVIEW, "whisper it") == (None, "whisper it")


def test_style_for_a_model_without_speech_metadata_is_refused_before_any_request(gemini_api):
    with pytest.raises(GeminiStyleUnsupportedError, match="does not accept a style"):
        synthesize_pcm("laughing", model=PREVIEW, style=STYLE)
    assert gemini_api.bodies == [] and gemini_api.prompts == []


# --- synthesize_pcm over the Interactions API --------------------------------------

def test_style_request_is_the_verified_interactions_shape(gemini_api):
    """Text verbatim, the direction only in the annotation, voice as a list,
    audio response format; the WAV (+ C2PA) answer comes back as its PCM."""
    gemini_api.interactions = [interaction_resp(gemini_lite_wav(PCM))]
    pcm = synthesize_pcm("laughing", voice="en-gb-tutor-11", model=LITE, style=STYLE)
    assert pcm == PCM
    assert gemini_api.bodies == [{
        "model": LITE,
        "input": [{"type": "user_input", "content": [{
            "type": "text", "text": "laughing",
            "annotations": [{"type": "speech_metadata", "style": STYLE}],
        }]}],
        "response_format": {"type": "audio"},
        "generation_config": {"speech_config": [{"voice": "en-gb-tutor-11"}]},
    }]
    assert gemini_api.prompts == []


def test_completed_interaction_without_audio_is_retried_and_counted(gemini_api):
    """Retries go through on_request too, so the sync runner's rate limiter
    counts them."""
    gemini_api.interactions = [interaction_resp(None), interaction_resp(gemini_lite_wav(PCM))]
    ticks = []
    assert synthesize_pcm("haha", model=LITE, style=STYLE,
                          on_request=lambda: ticks.append(1)) == PCM
    assert len(gemini_api.bodies) == len(ticks) == 2


def test_failed_interaction_is_not_retried(gemini_api):
    gemini_api.interactions = [interaction_resp(None, status="failed",
                                                errors=[{"message": "blocked"}]),
                               interaction_resp(gemini_lite_wav(PCM))]
    with pytest.raises(GeminiNoAudioError, match="status=failed.*blocked"):
        synthesize_pcm("haha", model=LITE, style=STYLE)
    assert len(gemini_api.bodies) == 1


# --- the Interactions transport (real _post_interaction, fake urlopen) -----------

class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def urlopen(monkeypatch):
    """Replace urllib's urlopen: records each request; answers with `reply`
    (bytes) or raises it (an exception)."""
    state = types.SimpleNamespace(requests=[], reply=json.dumps(interaction_resp(PCM)).encode())

    def fake(req, timeout=None):
        state.requests.append(req)
        if isinstance(state.reply, Exception):
            raise state.reply
        return _Resp(state.reply)

    monkeypatch.setattr(gemini_tts.urllib.request, "urlopen", fake)
    monkeypatch.setenv("GEMINI_API_KEY", "key-from-env")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    return state


def test_post_interaction_sends_the_body_to_v1beta_with_the_key(urlopen):
    body = gemini_tts.build_interaction_body("laughing", "Kore", STYLE, FLASH)
    assert gemini_tts._post_interaction(body) == interaction_resp(PCM)
    [req] = urlopen.requests
    assert req.full_url == "https://generativelanguage.googleapis.com/v1beta/interactions"
    assert req.get_header("X-goog-api-key") == "key-from-env"
    assert json.loads(req.data) == body


def test_post_interaction_errors_carry_the_status(urlopen, monkeypatch):
    """HTTP error -> its status (and Google's message); no response -> a
    network error (transient for the runner); no key -> nothing sent."""
    detail = b'{"error":{"message":"Speech metadata is not supported for this model.","code":"invalid_request"}}'
    urlopen.reply = urllib.error.HTTPError("https://x", 400, "Bad Request", {}, io.BytesIO(detail))
    with pytest.raises(GeminiAPIError, match="400: Speech metadata is not supported") as exc:
        gemini_tts._post_interaction({})
    assert exc.value.code == 400

    urlopen.reply = urllib.error.URLError("Name or service not known")
    with pytest.raises(GeminiNetworkError, match="unreachable"):
        gemini_tts._post_interaction({})

    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(GeminiAPIError, match="GEMINI_API_KEY"):
        gemini_tts._post_interaction({})
    assert len(urlopen.requests) == 2


# --- server: /v1/speak, GEMINI_TTS_STYLE, the studio ------------------------------

def _phrase_rows(paths):
    m = Manifest(str(paths.db), table="phrases", text_col="text")
    try:
        return [dict(r) for r in m.iter_rows()]
    finally:
        m.close()


@pytest.fixture
def server(bank_env, gemini_api, monkeypatch):
    """The app over an empty tmp bank with the real synthesize_pcm (faked
    transports); `server.client(model)` pins GEMINI_TTS_MODEL."""
    import src.server as srv

    def client(model):
        monkeypatch.setattr(srv, "GEMINI_TTS_MODEL", model)
        return TestClient(srv.app)
    return types.SimpleNamespace(client=client, api=gemini_api, paths=bank_env)


def test_speak_instruct_on_a_38_model_is_sent_as_style_and_never_saved(server):
    with server.client(LITE) as c:
        r = c.post("/v1/speak", data={"text": "laughing", "voice": "Puck",
                                      "instruct": "whisper it"})
    assert r.status_code == 200, r.text
    assert r.headers["x-source"] == "gemini" and r.content[:4] == b"OggS"
    assert _sent(server.api) == [("style", "laughing", "whisper it")]
    assert _phrase_rows(server.paths) == []


def test_tts_style_applies_to_misses_and_is_saved_as_provenance(server, monkeypatch):
    monkeypatch.setenv("GEMINI_TTS_STYLE", STYLE)
    with server.client(FLASH) as c:
        miss = c.post("/v1/speak", data={"text": "haha", "voice": "Kore"})
        hit = c.post("/v1/speak", data={"text": "haha", "voice": "Kore"})
        one_off = c.post("/v1/speak", data={"text": "haha", "voice": "Kore",
                                            "instruct": "whisper it"})
    assert miss.headers["x-source"] == "gemini" and hit.headers["x-source"] == "bank"
    assert one_off.status_code == 200
    # the one-off's instruct replaces the deployment style; it is not combined
    assert _sent(server.api) == [("style", "haha", STYLE), ("style", "haha", "whisper it")]
    [row] = _phrase_rows(server.paths)
    assert (row["status"], row["voice"], row["model"], row["style"]) == \
        (DONE, "gemini-Kore", FLASH, STYLE)


def test_tts_style_with_a_model_that_cannot_take_it(server, monkeypatch, caplog):
    """Startup warns; a miss is a 500 naming the setting, with nothing sent or
    saved; bank hits and instruct one-offs (which don't use it) still work."""
    from tests.test_speak import _bank_phrase
    banked = _bank_phrase(server.paths, "hello there", "gemini-Kore")
    monkeypatch.setenv("GEMINI_TTS_STYLE", STYLE)
    with caplog.at_level(logging.WARNING, logger="uvicorn.error"), server.client(PREVIEW) as c:
        miss = c.post("/v1/speak", data={"text": "haha"})
        hit = c.post("/v1/speak", data={"text": "hello there"})
        one_off = c.post("/v1/speak", data={"text": "haha", "instruct": "whisper it"})
    assert f"GEMINI_TTS_STYLE is set but GEMINI_TTS_MODEL={PREVIEW}" in caplog.text
    assert miss.status_code == 500 and "GEMINI_TTS_STYLE" in miss.json()["detail"]
    assert hit.headers["x-source"] == "bank" and hit.content == banked
    assert one_off.status_code == 200
    assert _sent(server.api) == [("generate", "whisper it: haha")]
    assert [r["text"] for r in _phrase_rows(server.paths)] == ["hello there"]


def test_studio_routes_style_by_model(server):
    with server.client(LITE) as c:
        on_38 = c.post("/v1/gemini/tts", data={"text": "laughing", "instruct": "warmly",
                                               "style": "slowly"})
        on_31 = c.post("/v1/gemini/tts", data={"text": "laughing", "style": "slowly",
                                               "model": PREVIEW})
    assert on_38.status_code == 200 and on_38.content[:4] == b"RIFF"
    assert on_31.status_code == 400 and "does not accept a style" in on_31.json()["detail"]
    assert _sent(server.api) == [("style", "laughing", "slowly. warmly")]
