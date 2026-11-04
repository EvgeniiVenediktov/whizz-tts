"""Synchronous single-phrase Gemini TTS → raw PCM. Used to author voice-clone
references (the UI "voice studio" and the `gemini-voice` CLI). Distinct from
`gemini_batch.py`, which builds the large async word bank.

Lazily imports google-genai; needs GEMINI_API_KEY in the environment.
"""
import io
import logging
import os
import time
import wave
from typing import Optional

from .constants import SAMPLE_RATE  # Gemini TTS emits 24 kHz / 16-bit mono PCM
DEFAULT_MODEL = "gemini-2.5-flash-preview-tts"

log = logging.getLogger(__name__)

# Gemini TTS sometimes returns a candidate with no content and finish_reason
# OTHER (no block reason, prompt-only usage) — non-deterministic, more often
# for text with long digit runs; the same call usually succeeds on a repeat.
# Only these reasons are retried; SAFETY/PROHIBITED_CONTENT/BLOCKLIST/SPII etc.
# and any prompt_feedback.block_reason fail on the first response.
_RETRYABLE_FINISH = {None, "FINISH_REASON_UNSPECIFIED", "OTHER"}
_RETRY_BACKOFF_S = 0.5   # sleep before retry n is n * this


class GeminiNoAudioError(RuntimeError):
    """Gemini answered without an audio part."""

# Prebuilt Gemini TTS voices: (name, style descriptor, gender). Voices are NOT
# language-specific — the model auto-detects language from the input text, so
# any voice speaks any supported language. The accent/language of a reference
# comes from the phrase you write (and optional `instruct`), not the voice.
# Gender is the community-perceived gender (Google doesn't officially assign one
# — the names are mythological); 14 female / 16 male.
# Source: https://ai.google.dev/gemini-api/docs/speech-generation
VOICES = [
    ("Zephyr", "Bright", "female"), ("Puck", "Upbeat", "male"),
    ("Charon", "Informative", "male"), ("Kore", "Firm", "female"),
    ("Fenrir", "Excitable", "male"), ("Leda", "Youthful", "female"),
    ("Orus", "Firm", "male"), ("Aoede", "Breezy", "female"),
    ("Callirrhoe", "Easy-going", "female"), ("Autonoe", "Bright", "female"),
    ("Enceladus", "Breathy", "male"), ("Iapetus", "Clear", "male"),
    ("Umbriel", "Easy-going", "male"), ("Algieba", "Smooth", "male"),
    ("Despina", "Smooth", "female"), ("Erinome", "Clear", "female"),
    ("Algenib", "Gravelly", "male"), ("Rasalgethi", "Informative", "male"),
    ("Laomedeia", "Upbeat", "female"), ("Achernar", "Soft", "female"),
    ("Alnilam", "Firm", "male"), ("Schedar", "Even", "male"),
    ("Gacrux", "Mature", "female"), ("Pulcherrima", "Forward", "female"),
    ("Achird", "Friendly", "male"), ("Zubenelgenubi", "Casual", "male"),
    ("Vindemiatrix", "Gentle", "female"), ("Sadachbia", "Lively", "male"),
    ("Sadaltager", "Knowledgeable", "male"), ("Sulafat", "Warm", "female"),
]

_VOICE_NAMES = {name.lower(): name for name, _, _ in VOICES}


def canonical_voice(name: str) -> Optional[str]:
    """Case-insensitive match against VOICES -> the canonical spelling ("puck"
    -> "Puck"), or None if it isn't a prebuilt voice. Banked clips are keyed by
    voice name, so normalizing here keeps "puck"/"Puck" one bank entry."""
    return _VOICE_NAMES.get(name.strip().lower())


_client = None


def _get_client():
    """Lazily construct and cache the google-genai client (one per process,
    like GeminiBatch._get_client) — synthesize_pcm is called per voice-studio
    HTTP request, so avoid reconnecting every time."""
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client()
    return _client


def _enum_name(value) -> Optional[str]:
    return getattr(value, "name", value) if value is not None else None


def _audio_or_reason(resp):
    """-> (pcm, None) when the response carries audio, else (None, why): a dict
    of finish_reason / finish_message / block_reason, whichever are present."""
    cand = (resp.candidates or [None])[0]
    parts = (cand.content.parts or []) if cand and cand.content else []
    for part in parts:
        if part.inline_data and part.inline_data.data:
            return part.inline_data.data, None
    feedback = getattr(resp, "prompt_feedback", None)
    why = {
        "finish_reason": _enum_name(cand.finish_reason) if cand else None,
        "finish_message": getattr(cand, "finish_message", None) if cand else None,
        "block_reason": _enum_name(feedback.block_reason) if feedback else None,
    }
    return None, why


def _empty_retries() -> int:
    return max(0, int(os.getenv("GEMINI_TTS_EMPTY_RETRIES", "2")))


def synthesize_pcm(text: str, voice: str = "Kore", instruct: str = "",
                   model: str = DEFAULT_MODEL) -> bytes:
    """Speak `text` with a Gemini prebuilt voice; return raw 24 kHz PCM bytes.

    `instruct` is style/accent direction (e.g. "British accent") — prepended,
    not spoken. A response with no audio and a transient finish_reason (OTHER,
    unset) is retried up to GEMINI_TTS_EMPTY_RETRIES times (default 2); a block
    is not. Raises GeminiNoAudioError, naming the reason, if no audio comes back.
    """
    from google.genai import types

    client = _get_client()
    prompt = f"{instruct}: {text}" if instruct else text
    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))),
    )
    attempts = 1 + _empty_retries()
    for attempt in range(1, attempts + 1):
        resp = client.models.generate_content(model=model, contents=prompt, config=config)
        pcm, why = _audio_or_reason(resp)
        if pcm:
            return pcm
        retryable = why["block_reason"] is None and why["finish_reason"] in _RETRYABLE_FINISH
        if not retryable or attempt == attempts:
            break
        log.warning("Gemini TTS returned no audio (finish_reason=%s, voice=%s); "
                    "retry %d/%d", why["finish_reason"], voice, attempt, attempts - 1)
        time.sleep(_RETRY_BACKOFF_S * attempt)
    detail = ", ".join(f"{k}={v}" for k, v in why.items() if v)
    raise GeminiNoAudioError(
        f"no audio in response ({detail or 'finish_reason=None'}) "
        f"after {attempt} attempt{'s' if attempt > 1 else ''}")


def pcm_to_wav_bytes(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV container (in memory) — no ffmpeg."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()
