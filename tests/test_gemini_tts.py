"""synthesize_pcm against a fake google-genai client: Gemini's transient
empty responses (finish_reason OTHER, no content) are retried, blocks are not,
and a final failure names the reason instead of an AttributeError."""
import types as pytypes

import pytest

genai_types = pytest.importorskip("google.genai.types")

import wordbank.gemini_tts as gemini_tts  # noqa: E402

PCM = b"\x00\x10" * 2400


def audio_resp():
    return genai_types.GenerateContentResponse(candidates=[genai_types.Candidate(
        finish_reason="STOP",
        content=genai_types.Content(parts=[genai_types.Part(
            inline_data=genai_types.Blob(data=PCM, mime_type="audio/L16;rate=24000"))]))])


def empty_resp(finish_reason="OTHER", finish_message=None):
    """What Gemini sent in the field: a candidate with no content."""
    return genai_types.GenerateContentResponse(candidates=[genai_types.Candidate(
        finish_reason=finish_reason, finish_message=finish_message, content=None)])


def prompt_blocked_resp():
    return genai_types.GenerateContentResponse(
        candidates=[],
        prompt_feedback=genai_types.GenerateContentResponsePromptFeedback(
            block_reason="PROHIBITED_CONTENT"))


@pytest.fixture
def fake_client(monkeypatch):
    """Install a client whose generate_content returns `responses` in order
    (the last one repeats); `calls` counts requests. No backoff sleep."""
    state = pytypes.SimpleNamespace(responses=[], calls=0)

    def generate_content(model, contents, config):
        state.calls += 1
        return state.responses[min(state.calls, len(state.responses)) - 1]

    client = pytypes.SimpleNamespace(models=pytypes.SimpleNamespace(generate_content=generate_content))
    monkeypatch.setattr(gemini_tts, "_get_client", lambda: client)
    monkeypatch.setattr(gemini_tts, "_RETRY_BACKOFF_S", 0)
    monkeypatch.delenv("GEMINI_TTS_EMPTY_RETRIES", raising=False)
    return state


def test_audio_on_first_call_returns_pcm(fake_client):
    fake_client.responses = [audio_resp()]
    assert gemini_tts.synthesize_pcm("hello", voice="Puck") == PCM
    assert fake_client.calls == 1


def test_empty_then_audio_succeeds_on_retry(fake_client):
    fake_client.responses = [empty_resp(), audio_resp()]
    assert gemini_tts.synthesize_pcm("Persistence check 1793813068", voice="Puck") == PCM
    assert fake_client.calls == 2


def test_missing_candidate_without_block_reason_is_retried(fake_client):
    fake_client.responses = [genai_types.GenerateContentResponse(candidates=[]), audio_resp()]
    assert gemini_tts.synthesize_pcm("hello") == PCM
    assert fake_client.calls == 2


def test_always_empty_raises_naming_finish_reason_after_n_plus_1_calls(fake_client):
    fake_client.responses = [empty_resp(finish_message="model gave up")]
    with pytest.raises(gemini_tts.GeminiNoAudioError) as exc:
        gemini_tts.synthesize_pcm("Order number 1793813068")
    assert fake_client.calls == 3                      # default 2 retries
    assert "finish_reason=OTHER" in str(exc.value)
    assert "finish_message=model gave up" in str(exc.value)
    assert "after 3 attempts" in str(exc.value)


def test_retry_count_comes_from_env(fake_client, monkeypatch):
    monkeypatch.setenv("GEMINI_TTS_EMPTY_RETRIES", "0")
    fake_client.responses = [empty_resp(), audio_resp()]
    with pytest.raises(gemini_tts.GeminiNoAudioError, match="after 1 attempt\\b"):
        gemini_tts.synthesize_pcm("hello")
    assert fake_client.calls == 1


@pytest.mark.parametrize("reason", ["SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"])
def test_blocked_finish_reason_is_not_retried(fake_client, reason):
    fake_client.responses = [empty_resp(reason), audio_resp()]
    with pytest.raises(gemini_tts.GeminiNoAudioError, match=f"finish_reason={reason}"):
        gemini_tts.synthesize_pcm("hello")
    assert fake_client.calls == 1


def test_prompt_block_reason_is_not_retried(fake_client):
    fake_client.responses = [prompt_blocked_resp(), audio_resp()]
    with pytest.raises(gemini_tts.GeminiNoAudioError, match="block_reason=PROHIBITED_CONTENT"):
        gemini_tts.synthesize_pcm("hello")
    assert fake_client.calls == 1


def test_speak_answers_502_with_the_reason_and_saves_nothing(bank_env, fake_client):
    """End to end through /v1/speak: the real synthesize_pcm (fake client
    only), so the 502 detail is the finish_reason, not an AttributeError."""
    from fastapi.testclient import TestClient
    from src.server import app
    from wordbank.manifest import Manifest

    fake_client.responses = [empty_resp()]
    with TestClient(app) as c:
        r = c.post("/v1/speak", data={"text": "Persistence check 1793813068", "voice": "Puck"})
    assert r.status_code == 502
    assert "finish_reason=OTHER" in r.json()["detail"]
    assert fake_client.calls == 3
    m = Manifest(str(bank_env.db), table="phrases", text_col="text")
    try:
        assert list(m.iter_rows()) == []
    finally:
        m.close()
