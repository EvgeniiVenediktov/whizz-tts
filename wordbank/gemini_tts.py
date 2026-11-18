"""Synchronous single-phrase Gemini TTS → raw PCM. Used by /v1/speak, the voice
studio, the `gemini-voice` CLI and the sync bank runner (`gemini_run.py`).
Distinct from `gemini_batch.py`, which builds the large async word bank.

Two request paths, picked per model by the capability registry (TTS_MODELS):
generateContent through google-genai (lazily imported) for plain text and for
inline directions, and the Interactions API (raw REST, stdlib) for a `style`
direction, which only the 3.8 models accept. Needs GEMINI_API_KEY (or
GOOGLE_API_KEY) in the environment.
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
# 3.8 Flash: with a speech_metadata style it said the words 3.8 Lite acted out
# ("laughing", "haha", "russian"), at 1.5x Lite's price. A 3.x model because the
# 2.5 TTS models answer imperative text ("Read this one slowly please.") with
# 400 "Model tried to generate text" and drop audio far more often on digit
# runs. The 3.8 models return a whole WAV file (with a C2PA provenance chunk),
# not bare PCM — pcm_from_payload unwraps it. NOT the word bank's model
# (gemini_batch.BATCH_DEFAULT_MODEL), so /v1/speak phrases can sound slightly
# different from the banked words.
DEFAULT_MODEL = "gemini-3.8-flash-tts"

# TTS (speechConfig), batches and file uploads are v1beta-only. google-genai 3.x
# defaults to v1, which rejects speechConfig with 400 INVALID_ARGUMENT, so every
# client pins the version explicitly (via new_client) instead of trusting the SDK.
GEMINI_API_VERSION = "v1beta"

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


# --- model capabilities: how a delivery direction reaches each TTS model -------
# Every caller (synthesize_pcm, the batch request builder, the CLI, the server)
# routes directions through this registry, so it's the one place to add a model.
#
# - style:  accepts a `style` direction as structured speech metadata (the
#           Interactions `speech_metadata` annotation, or `speechMetadata` on a
#           generateContent part). The text stays a verbatim transcript.
# - inline: treats a leading "<direction>: " in the text as direction, not as
#           words to speak (today's `instruct` prefix).
# Verified live (2026-11): both 3.8 models take a style and read any inline
# prefix ALOUD ("Say …: word" came back as the whole prompt spoken); both reject
# generateContent system_instruction ("Developer instruction is not enabled").
# 3.1 rejects speech metadata (400 "Speech metadata is not supported for this
# model."). 2.5 was not probed for style; it is treated like 3.1.
#
# - locale_voices: takes the voice library's locale voices (`en-gb-tutor-11`),
#           not just the 30 classic ones. Verified (2026-11) on both 3.8 models
#           with en-gb-tutor-11; UNVERIFIED on 3.1/2.5 (never tried), so refused
#           there before any request (check_voice). Classic voices work on all.
@dataclass(frozen=True)
class TTSModel:
    style: bool
    inline: bool
    locale_voices: bool


TTS_MODELS = {
    "gemini-3.8-flash-tts": TTSModel(style=True, inline=False, locale_voices=True),
    "gemini-3.8-flash-lite-tts": TTSModel(style=True, inline=False, locale_voices=True),
    "gemini-3.1-flash-tts-preview": TTSModel(style=False, inline=True, locale_voices=False),
    "gemini-2.5-flash-preview-tts": TTSModel(style=False, inline=True, locale_voices=False),
    "gemini-2.5-pro-preview-tts": TTSModel(style=False, inline=True, locale_voices=False),
}
# An unlisted model keeps the pre-registry behaviour (inline prefix, no style,
# classic voices only).
_UNLISTED = TTSModel(style=False, inline=True, locale_voices=False)


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
    if instruct and not caps.inline:
        raise GeminiStyleUnsupportedError(
            f"model {model!r} takes no delivery direction (neither style nor inline)")
    return None, instruct

# The 30 classic prebuilt Gemini TTS voices: (name, style descriptor, gender).
# Voices are NOT language-specific — the model auto-detects language from the
# input text, so any voice speaks any supported language. Every TTS model takes
# them. Gender is the community-perceived one, kept for /v1/gemini/voices
# (14 female / 16 male); the voice library (GET /v1beta/voices) now lists a
# gender too and agrees except Pulcherrima ("neutral" there).
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

# Locale voices the server accepts beyond the classic ones: the curated
# per-language picks (languages.VOICE_LANGUAGES; its en-us picks are classic).
# Labelled by their lowercase library id ("en-gb-tutor-11"), as the batch path
# files them (`<lang>/gemini-<id>/…`). 3.8 models only (check_voice).
LOCALE_VOICES = frozenset(v["id"] for lang in VOICE_LANGUAGES
                          for v in lang["voices"]) - _VOICE_NAMES.keys()


def canonical_voice(name: str) -> Optional[str]:
    """Case-insensitive match against the classic VOICES and LOCALE_VOICES ->
    the canonical label, or None if it is neither. Classic voices keep their
    capitalized name ("puck" -> "Puck": banks key on it, `gemini-Kore`); a
    locale voice is its lowercase id ("EN-GB-Tutor-11" -> "en-gb-tutor-11").
    Banked clips are keyed by this label, so "puck"/"Puck" stay one entry."""
    key = name.strip().lower()
    return _VOICE_NAMES.get(key) or (key if key in LOCALE_VOICES else None)


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
    return genai.Client(http_options={"api_version": GEMINI_API_VERSION})


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
# Raw REST over urllib rather than google-genai's client.interactions: the
# request/response below is the wire format verified live, while the SDK's
# generated Interactions surface has changed between the releases this project
# runs (2.31 in the dev venv, 3.x in the images). The URL carries
# GEMINI_API_VERSION like every google-genai client from new_client().
GEMINI_API_BASE = "https://generativelanguage.googleapis.com"
_INTERACTIONS_TIMEOUT_S = 120
# A completed interaction with no audio is the Interactions analogue of an
# empty generateContent candidate (finish_reason OTHER) and is retried; a
# failed / cancelled / incomplete one, or one carrying errors, is not.
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
        with urllib.request.urlopen(req, timeout=_INTERACTIONS_TIMEOUT_S) as resp:
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
    """Speak `text` with a Gemini voice; return raw 24 kHz PCM bytes (a WAV
    payload is unwrapped — see pcm_from_payload).

    Directions are routed per model (route_direction):
    - `style` (delivery direction): only for style-capable models (3.8), sent as
      a speech_metadata annotation through the Interactions API with `text`
      verbatim. Any other model raises GeminiStyleUnsupportedError before a
      request is made.
    - `instruct` (style/accent direction, e.g. "British accent"): on
      style-capable models it is sent as the style (after `style` when both are
      given, see combine_style) — never inline, since 3.8 reads inline text
      aloud. On other models it is prepended to the text as today.
    - neither: generateContent with the bare text, as before.

    A response with no audio and a transient reason (generateContent
    finish_reason OTHER/unset; an Interactions status "completed") is retried
    up to GEMINI_TTS_EMPTY_RETRIES times (default 2); a block or failure is not.
    Raises GeminiNoAudioError, naming the reason, if no audio comes back, and
    GeminiAudioFormatError (not retried) for audio in another format.
    `on_request` is called before every request, retries included (the sync
    bank runner passes its rate limiter). A locale voice on a model not
    verified with it raises GeminiVoiceUnsupportedError first (check_voice).
    """
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
