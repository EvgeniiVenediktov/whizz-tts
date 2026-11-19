"""Synchronous single-phrase Gemini TTS -> raw PCM, for /v1/speak, the studio
and `gemini-run` (gemini_batch.py builds the async word bank). Per model
(TTS_MODELS): generateContent via google-genai (lazy) for plain text and inline
directions, the Interactions API (stdlib REST) for a `style` (3.8 only).
Needs GEMINI_API_KEY (or GOOGLE_API_KEY).
"""
import base64
import io
import json
import logging
import os
import socket
import struct
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from .constants import SAMPLE_RATE  # Gemini TTS emits 24 kHz / 16-bit mono PCM
from .languages import VOICE_LANGUAGES
# Not the word bank's model (gemini_batch.BATCH_DEFAULT_MODEL); 2.5 rejects
# imperative text and drops audio on digit runs (AGENTS.md, batch notes).
DEFAULT_MODEL = "gemini-3.8-flash-tts"

# TTS, batches and uploads are v1beta-only; google-genai 3.x defaults to v1
# (400 on speechConfig), so new_client pins it.
GEMINI_API_VERSION = "v1beta"
_HTTP_TIMEOUT_S = 120     # every Gemini request (a server thread waits on it)

log = logging.getLogger(__name__)

# An empty candidate with finish_reason OTHER is transient (a repeat usually
# works); blocks (SAFETY, PROHIBITED_CONTENT, a block_reason ...) are not retried.
_RETRYABLE_FINISH = {None, "FINISH_REASON_UNSPECIFIED", "OTHER"}
_RETRY_BACKOFF_S = 0.5   # sleep before retry n is n * this


class GeminiNoAudioError(RuntimeError):
    """Gemini answered without an audio part."""


class GeminiAudioFormatError(ValueError):
    """Gemini's audio isn't 16-bit mono PCM at SAMPLE_RATE (never resampled)."""


class GeminiStyleUnsupportedError(ValueError):
    """A `style` direction was asked of a model that doesn't accept one. Raised
    before any request is sent."""


class GeminiVoiceUnsupportedError(ValueError):
    """A locale voice (any voice beyond the 30 classic ones) was asked of a
    model it isn't verified on (TTSModel.locale_voices). Raised before any
    request is sent."""


class GeminiAPIError(RuntimeError):
    """An Interactions request failed. `code` is the HTTP status, like
    google-genai's APIError.code, so callers classify both request paths the
    same way; None when there was no response (GeminiNetworkError) or no key."""

    def __init__(self, message: str, code: Optional[int] = None):
        super().__init__(message)
        self.code = code


class GeminiNetworkError(GeminiAPIError):
    """No HTTP response: the connection failed or timed out (transient)."""


# --- model capabilities: the one place a TTS model is described ---------------
# style: takes a direction as speech metadata, the text verbatim; such a model
#   reads an inline prefix ALOUD, so it never gets one (the others get only the
#   inline prefix). locale_voices: verified with the library's locale voices
#   (`en-gb-tutor-11`); refused elsewhere before any request (check_voice).
@dataclass(frozen=True)
class TTSModel:
    style: bool
    locale_voices: bool


TTS_MODELS = {
    "gemini-3.8-flash-tts": TTSModel(style=True, locale_voices=True),
    "gemini-3.8-flash-lite-tts": TTSModel(style=True, locale_voices=True),
    "gemini-3.1-flash-tts-preview": TTSModel(style=False, locale_voices=False),
    "gemini-2.5-flash-preview-tts": TTSModel(style=False, locale_voices=False),
    "gemini-2.5-pro-preview-tts": TTSModel(style=False, locale_voices=False),
}
_UNLISTED = TTSModel(style=False, locale_voices=False)


def model_caps(model: str) -> TTSModel:
    """Capabilities of a TTS model id ("models/" prefix allowed)."""
    return TTS_MODELS.get(model.removeprefix("models/"), _UNLISTED)


def supports_style(model: str) -> bool:
    return model_caps(model).style


def style_models() -> list:
    return sorted(m for m, caps in TTS_MODELS.items() if caps.style)


def locale_voice_models() -> list:
    return sorted(m for m, caps in TTS_MODELS.items() if caps.locale_voices)


def combine_style(style: Optional[str], instruct: Optional[str]) -> Optional[str]:
    """One style string from a base `style` and a per-call `instruct`: the
    style first, then the instruct as its own sentence ("plain and neutral" +
    "whisper it" -> "plain and neutral. whisper it"). Either alone is returned
    unchanged; neither -> None."""
    parts = [p.strip() for p in (style, instruct) if p and p.strip()]
    if len(parts) < 2:
        return parts[0] if parts else None
    first = parts[0] if parts[0][-1] in ".!?;:" else parts[0] + "."
    return f"{first} {parts[1]}"


def route_direction(model: str, instruct: Optional[str] = None,
                    style: Optional[str] = None) -> Tuple[Optional[str], str]:
    """How a delivery direction reaches `model` -> (style, inline_prefix); at
    most one is non-empty. Style-capable models get `style` and `instruct`
    combined (combine_style) as the style, never inline: they would speak it.
    Other models keep `instruct` as the inline prefix; a `style` for them, or
    any direction for a model that takes neither, raises
    GeminiStyleUnsupportedError."""
    caps = model_caps(model)
    style = (style or "").strip() or None
    instruct = (instruct or "").strip()
    if caps.style:
        return combine_style(style, instruct), ""
    if style:
        raise GeminiStyleUnsupportedError(
            f"model {model!r} does not accept a style direction (speech metadata); "
            f"style-capable models: {', '.join(style_models())}")
    return None, instruct


# The 30 classic voices (name, style, gender), on every model and in any
# language (detected from the text). ai.google.dev/gemini-api/docs/speech-generation
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

# The curated locale voices (languages.VOICE_LANGUAGES), labelled by their
# lowercase library id ("en-gb-tutor-11"). 3.8 models only (check_voice).
LOCALE_VOICES = frozenset(v["id"] for lang in VOICE_LANGUAGES
                          for v in lang["voices"]) - _VOICE_NAMES.keys()


def canonical_voice(name: str) -> Optional[str]:
    """Case-insensitive match against the classic VOICES and LOCALE_VOICES ->
    the canonical label, or None if it is neither. Classic voices keep their
    capitalized name ("puck" -> "Puck": banks key on it, words and phrases); a
    locale voice is its lowercase id ("EN-GB-Tutor-11" -> "en-gb-tutor-11").
    Banked clips are keyed by this label, so "puck"/"Puck" stay one entry."""
    key = name.strip().lower()
    return _VOICE_NAMES.get(key) or (key if key in LOCALE_VOICES else None)


def voice_label(name: str) -> str:
    """The bank label for any voice name: canonical_voice's, else the lowercase
    id ("kore" -> "Kore"). Also reads the legacy phrase spelling "gemini-Kore"
    (older banks and upstreams) as "Kore"."""
    name = name.strip().removeprefix("gemini-")
    return canonical_voice(name) or name.lower()


def is_classic_voice(name: str) -> bool:
    return name.strip().lower() in _VOICE_NAMES


def check_voice(voice: str, model: str) -> None:
    """Raise GeminiVoiceUnsupportedError when `voice` is not a classic voice and
    `model` isn't verified with locale voices (TTSModel.locale_voices). Any
    non-classic name counts, curated or not: the batch CLI passes ids through."""
    if model_caps(model).locale_voices or is_classic_voice(voice):
        return
    raise GeminiVoiceUnsupportedError(
        f"voice {voice!r} is not one of the 30 classic voices; locale voices are "
        f"verified only on {', '.join(locale_voice_models())}, not on {model}")


_client = None


def new_client():
    """A google-genai client pinned to GEMINI_API_VERSION. Every Gemini client
    in the project (TTS, batch, scripts) is built here."""
    from google import genai
    return genai.Client(http_options={"api_version": GEMINI_API_VERSION,
                                      "timeout": _HTTP_TIMEOUT_S * 1000})   # ms


def _get_client():
    """Lazily construct and cache the google-genai client (one per process,
    like GeminiBatch._get_client) — synthesize_pcm is called per voice-studio
    HTTP request, so avoid reconnecting every time."""
    global _client
    if _client is None:
        _client = new_client()
    return _client


def _enum_name(value) -> Optional[str]:
    return getattr(value, "name", value) if value is not None else None


def _mime_rate(mime_type: Optional[str]) -> Optional[int]:
    """The rate= parameter of a mime type ("audio/L16;codec=pcm;rate=24000")."""
    for param in (mime_type or "").split(";")[1:]:
        key, _, value = param.partition("=")
        if key.strip().lower() == "rate":
            try:
                return int(value.strip())
            except ValueError:
                raise GeminiAudioFormatError(
                    f"unreadable rate in Gemini audio mime type {mime_type!r}") from None
    return None


def _wav_data_chunk(wav: bytes) -> bytes:
    """The `data` chunk of a RIFF/WAVE payload, once its `fmt ` checks out as
    16-bit mono PCM at SAMPLE_RATE. Chunks are walked by their own sizes — the
    RIFF size field isn't trusted, a size running past the payload (streamed
    WAVs write 0xFFFFFFFF) is clamped to it, unknown chunks (C2PA, LIST) are
    skipped."""
    if wav[8:12] != b"WAVE":
        raise GeminiAudioFormatError(f"Gemini audio is a RIFF {wav[8:12]!r} file, not WAVE")
    fmt = data = None
    pos = 12
    while pos + 8 <= len(wav):
        cid = wav[pos:pos + 4]
        size, = struct.unpack_from("<I", wav, pos + 4)
        body = wav[pos + 8:pos + 8 + size]
        if cid == b"fmt " and fmt is None:
            fmt = body
        elif cid == b"data" and data is None:
            data = body
        pos += 8 + size + (size & 1)        # odd-sized chunks carry a pad byte
    if fmt is None or len(fmt) < 16 or data is None:
        raise GeminiAudioFormatError(
            f"Gemini WAV audio ({len(wav)} bytes) has no complete fmt + data chunk")
    tag, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", fmt)
    if (tag, channels, rate, bits) != (1, 1, SAMPLE_RATE, 16):
        raise GeminiAudioFormatError(
            f"Gemini WAV audio is format {tag}, {channels} ch, {rate} Hz, {bits}-bit; "
            f"expected PCM (1), mono, {SAMPLE_RATE} Hz, 16-bit")
    return data


def pcm_from_payload(payload: bytes, mime_type: Optional[str] = None) -> bytes:
    """A Gemini audio payload (inline_data.data + its mime_type) -> raw 16-bit
    mono PCM at SAMPLE_RATE, the format every caller writes or encodes.

    Models differ: gemini-3.1-flash-tts-preview sends bare PCM
    ("audio/L16;codec=pcm;rate=24000"), passed through unchanged;
    gemini-3.8-flash-lite-tts sends a whole RIFF/WAVE file (fmt, data, then a
    ~7.6 KB C2PA provenance chunk). Played as PCM, its header is a click and
    the C2PA chunk ~0.15 s of noise, so only the data chunk is kept. Any other
    rate / width / channel count raises GeminiAudioFormatError naming what
    arrived — it is never resampled.
    """
    rate = _mime_rate(mime_type)
    if rate is not None and rate != SAMPLE_RATE:
        raise GeminiAudioFormatError(
            f"Gemini audio is {mime_type!r}; expected rate={SAMPLE_RATE}")
    if payload[:4] == b"RIFF":
        return _wav_data_chunk(payload)
    return payload


def _audio_or_reason(resp):
    """-> (pcm, None) when the response carries audio, else (None, why): a dict
    of finish_reason / finish_message / block_reason, whichever are present.
    Raises GeminiAudioFormatError for audio pcm_from_payload can't convert."""
    cand = (resp.candidates or [None])[0]
    parts = (cand.content.parts or []) if cand and cand.content else []
    for part in parts:
        if part.inline_data and part.inline_data.data:
            pcm = pcm_from_payload(part.inline_data.data, part.inline_data.mime_type)
            if pcm:                         # a WAV with an empty data chunk is no audio
                return pcm, None
    feedback = getattr(resp, "prompt_feedback", None)
    why = {
        "finish_reason": _enum_name(cand.finish_reason) if cand else None,
        "finish_message": getattr(cand, "finish_message", None) if cand else None,
        "block_reason": _enum_name(feedback.block_reason) if feedback else None,
    }
    return None, why


def _empty_retries() -> int:
    return max(0, int(os.getenv("GEMINI_TTS_EMPTY_RETRIES", "2")))


# --- Interactions API (style directions) --------------------------------------
# Raw REST, not client.interactions: this wire format is verified live, the
# SDK's surface changed between releases.
GEMINI_API_BASE = "https://generativelanguage.googleapis.com"
# "completed" without audio is retried like finish_reason OTHER; failed /
# cancelled / incomplete, or with errors, is not.
_RETRYABLE_STATUS = {None, "completed"}


def build_interaction_body(text: str, voice: str, style: str, model: str) -> dict:
    """Single-speaker Interactions request: `text` is the verbatim transcript,
    `style` rides in a speech_metadata annotation (never in the text)."""
    return {
        "model": model,
        "input": [{"type": "user_input", "content": [{
            "type": "text", "text": text,
            "annotations": [{"type": "speech_metadata", "style": style}],
        }]}],
        "response_format": {"type": "audio"},
        "generation_config": {"speech_config": [{"voice": voice}]},
    }


def _api_key() -> Optional[str]:
    # google-genai's precedence: GOOGLE_API_KEY wins when both are set
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or None


def _error_detail(raw: bytes) -> str:
    """{"error": {"message": ...}} -> the message, else the raw text."""
    text = raw.decode("utf-8", errors="replace").strip()
    try:
        err = json.loads(text).get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
    except (ValueError, AttributeError):
        pass
    return text[:500]


def _post_interaction(body: dict) -> dict:
    """POST one Interactions request -> the parsed JSON response. The single
    transport of the style path (tests replace it). HTTP errors raise
    GeminiAPIError with the status; no response raises GeminiNetworkError."""
    key = _api_key()
    if not key:
        raise GeminiAPIError("no Gemini API key: set GEMINI_API_KEY (or GOOGLE_API_KEY)")
    req = urllib.request.Request(
        f"{GEMINI_API_BASE}/{GEMINI_API_VERSION}/interactions",
        data=json.dumps(body).encode("utf-8"),
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise GeminiAPIError(
            f"Gemini Interactions {exc.code}: {_error_detail(exc.read()) or exc.reason}",
            code=exc.code) from exc
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as exc:
        raise GeminiNetworkError(
            f"Gemini Interactions unreachable: {getattr(exc, 'reason', exc)}") from exc


def _interaction_audio_or_reason(resp: dict):
    """-> (pcm, None) from the last audio block of the model's output steps,
    else (None, why): the interaction's status and errors."""
    for step in reversed(resp.get("steps") or []):
        if step.get("type") != "model_output":
            continue
        for item in reversed(step.get("content") or []):
            if item.get("type") == "audio" and item.get("data"):
                pcm = pcm_from_payload(base64.b64decode(item["data"]), item.get("mime_type"))
                if pcm:                     # a WAV with an empty data chunk is no audio
                    return pcm, None
    return None, {"status": resp.get("status"), "errors": resp.get("errors") or None}


def synthesize_pcm(text: str, voice: str = "Kore", instruct: str = "",
                   model: str = DEFAULT_MODEL, style: Optional[str] = None,
                   on_request: Optional[Callable[[], None]] = None) -> bytes:
    """Speak `text` in a Gemini voice -> raw 24 kHz PCM (pcm_from_payload).
    `instruct`/`style` are routed per model (route_direction): a style-capable
    model gets them as the Interactions speech_metadata style, `text` verbatim;
    another gets `instruct` as an inline prefix, and a `style` raises
    GeminiStyleUnsupportedError before any request. No audio for a transient
    reason is retried GEMINI_TTS_EMPTY_RETRIES times (default 2), then
    GeminiNoAudioError names the reason. `on_request` runs before every request
    (gemini-run's rate limiter)."""
    check_voice(voice, model)
    style, inline = route_direction(model, instruct, style)
    if style:
        body = build_interaction_body(text, voice, style, model)

        def fetch():
            pcm, why = _interaction_audio_or_reason(_post_interaction(body))
            return pcm, why, not why or (why["errors"] is None
                                         and why["status"] in _RETRYABLE_STATUS)
    else:
        from google.genai import types

        client = _get_client()
        prompt = f"{inline}: {text}" if inline else text
        config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))),
        )

        def fetch():
            resp = client.models.generate_content(model=model, contents=prompt, config=config)
            pcm, why = _audio_or_reason(resp)
            return pcm, why, not why or (why["block_reason"] is None
                                         and why["finish_reason"] in _RETRYABLE_FINISH)

    attempts = 1 + _empty_retries()
    for attempt in range(1, attempts + 1):
        if on_request is not None:
            on_request()
        pcm, why, retryable = fetch()
        if pcm:
            return pcm
        if not retryable or attempt == attempts:
            break
        log.warning("Gemini TTS returned no audio (%s, voice=%s); retry %d/%d",
                    _reason(why) or "no reason", voice, attempt, attempts - 1)
        time.sleep(_RETRY_BACKOFF_S * attempt)
    raise GeminiNoAudioError(
        f"no audio in response ({_reason(why) or 'finish_reason=None'}) "
        f"after {attempt} attempt{'s' if attempt > 1 else ''}")


def _reason(why: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in why.items() if v)


def pcm_to_wav_bytes(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV container (in memory) — no ffmpeg."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()
