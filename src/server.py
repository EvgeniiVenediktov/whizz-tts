import asyncio
import logging
import os
import subprocess
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Optional

import jwt
from fastapi import Depends, FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from .auth import require_auth
from .upstream import UpstreamError, fetch_speech
from wordbank.constants import BANK_OGG_APPLICATION, BANK_OGG_BITRATE
from wordbank.gemini_tts import (DEFAULT_MODEL as GEMINI_TTS_DEFAULT_MODEL, TTS_MODELS,
                                 GeminiNoAudioError, GeminiStyleUnsupportedError,
                                 GeminiVoiceUnsupportedError, canonical_voice, check_voice,
                                 is_classic_voice, style_models, supports_style, voice_label)
from wordbank.languages import normalize_lang
from wordbank.lookup import BankLookup
from wordbank.paths import voice_from_path

# NOTE: src.audio (ffmpeg) and google-genai are imported LAZILY inside the
# handlers, so the serving image needs neither at import time.

MAX_TEXT_LEN = int(os.getenv("MAX_TEXT_LEN", "2000"))
MAX_DIRECTION_LEN = 500           # instruct / style
MAX_BODY_BYTES = 64 * 1024        # every POST is a small form
MAX_RESULTS = 50                  # /v1/search, /v1/bank
FRONTEND_DIR = os.getenv("FRONTEND_DIR", "frontend")
DEFAULT_LANG = os.getenv("DEFAULT_LANG", "en-us")
GEMINI_TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", GEMINI_TTS_DEFAULT_MODEL)
GEMINI_TTS_VOICE = os.getenv("GEMINI_TTS_VOICE", "Kore")
SPEAK_UPSTREAM_TIMEOUT = float(os.getenv("SPEAK_UPSTREAM_TIMEOUT", "60"))

log = logging.getLogger("uvicorn.error")


def _truthy(val: Optional[str]) -> bool:
    return (val or "").lower() not in ("", "0", "false", "no", "off")


def _gemini_configured() -> bool:
    """google-genai authenticates from GEMINI_API_KEY (or GOOGLE_API_KEY). With
    one set, /v1/speak can fill bank misses, so the service can serve even
    without a bank. Read per call, like JWT_SECRET."""
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def _upstream_url() -> Optional[str]:
    """Edge mode: with SPEAK_UPSTREAM_URL set, /v1/speak misses are fetched
    from that whizz-tts server (and saved here) instead of from Gemini."""
    return os.getenv("SPEAK_UPSTREAM_URL") or None


def _tts_style() -> Optional[str]:
    """GEMINI_TTS_STYLE: a deployment-wide delivery direction for every clip
    /v1/speak generates for the bank (misses, saved or with SPEAK_STORE=0
    handed to edges), sent as the speech_metadata style. Not applied to
    `instruct` one-offs or the studio. Needs a style-capable GEMINI_TTS_MODEL.
    Read per call; unset by default."""
    return os.getenv("GEMINI_TTS_STYLE", "").strip() or None


def _warn_about_tts_style(log: logging.Logger) -> None:
    """GEMINI_TTS_STYLE with a model that can't take it would fail every miss
    (500); say so at startup rather than on the first request."""
    if not _tts_style():
        return
    if _upstream_url():
        log.warning("GEMINI_TTS_STYLE is ignored in edge mode (SPEAK_UPSTREAM_URL): "
                    "misses come from the upstream, generated with its own style")
    elif not supports_style(GEMINI_TTS_MODEL):
        log.warning(f"GEMINI_TTS_STYLE is set but GEMINI_TTS_MODEL={GEMINI_TTS_MODEL} does "
                    f"not accept a style (style-capable: {', '.join(style_models())}): every "
                    "/v1/speak miss will fail with 500 until one of them changes")


def _speak_store() -> bool:
    """SPEAK_STORE=0: /v1/speak never writes the bank — a stateless Gemini
    front for edges, which hold the clips. Hits from an existing bank are still
    served (reads aren't storage). Default on. Read per call."""
    return os.getenv("SPEAK_STORE", "1").strip().lower() not in ("0", "false", "no", "off")


def _warn_about_upstream_token(log: logging.Logger) -> None:
    """Edge mode authenticates upstream with a static token; once it lapses
    every miss fails, so flag a missing, unreadable, or expiring one at startup."""
    if not _upstream_url():
        return
    token = os.getenv("SPEAK_UPSTREAM_TOKEN")
    if not token:
        log.warning("SPEAK_UPSTREAM_URL is set but SPEAK_UPSTREAM_TOKEN is not: "
                    "an upstream with JWT_SECRET will reject every miss")
        return
    try:
        exp = jwt.decode(token, options={"verify_signature": False}).get("exp")
    except jwt.InvalidTokenError:
        log.warning("SPEAK_UPSTREAM_TOKEN is not a readable JWT")
        return
    if exp is not None and exp - time.time() < 30 * 86400:
        days = (exp - time.time()) / 86400
        log.warning(f"SPEAK_UPSTREAM_TOKEN {'expired' if days < 0 else 'expires'} "
                    f"in {abs(days):.0f} days: renew it with scripts/make_edge_env.py")


def _gemini_voice(voice: Optional[str]) -> Optional[str]:
    """Canonical voice label for a request field (a classic name like "Kore",
    or a curated locale voice id like "en-gb-tutor-11"), None when omitted. An
    unknown name is a 400 here rather than a 502 from a wasted Gemini call."""
    if not voice or not voice.strip():
        return None
    canon = canonical_voice(voice)
    if canon is None:
        raise HTTPException(400, f"unknown Gemini voice {voice!r} "
                                 "(see /v1/gemini/voices and /v1/gemini/languages)")
    return canon


def _bank_voice(voice: Optional[str]) -> Optional[str]:
    """A lookup's `voice` as the bank label (voice_label: "kore" and the legacy
    "gemini-Kore" -> "Kore"), None when omitted. Lenient, unlike _gemini_voice:
    a read can't waste a Gemini call, and a bank may hold any library voice."""
    return voice_label(voice) if voice and voice.strip() else None


def _default_voice() -> str:
    """GEMINI_TTS_VOICE as a bank label: preferred by lookups without a voice."""
    return canonical_voice(GEMINI_TTS_VOICE) or GEMINI_TTS_VOICE


def _check_voice(voice: str, model: str, server_default: bool = False) -> None:
    """A locale voice on a model not verified with it (only the 3.8 models
    are) fails here, before any Gemini call: 400 for a voice the request chose,
    500 for the server's own GEMINI_TTS_VOICE (a misconfiguration)."""
    try:
        check_voice(voice, model)
    except GeminiVoiceUnsupportedError as exc:
        if server_default:
            raise HTTPException(500, f"server misconfigured: GEMINI_TTS_VOICE: {exc}") from exc
        raise HTTPException(400, str(exc)) from exc


def _lang(lang: str) -> str:
    """Canonical bank language code for a request field (trimmed, lowercased),
    or a 400. `lang` is the first directory of every saved clip, so every route
    taking it runs this before any lookup, synthesis, upstream fetch, or write:
    "../x" or an absolute path never gets near the filesystem."""
    try:
        return normalize_lang(lang)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _direction(value: Optional[str], name: str) -> str:
    """A stripped instruct/style field, or 400 when it exceeds MAX_DIRECTION_LEN."""
    value = (value or "").strip()
    if len(value) > MAX_DIRECTION_LEN:
        raise HTTPException(400, f"{name} exceeds {MAX_DIRECTION_LEN} characters")
    return value


def _cors_origins() -> list:
    """Comma-separated CORS_ORIGINS env var -> origin list; unset -> ["*"]."""
    raw = os.getenv("CORS_ORIGINS")
    if not raw:
        return ["*"]
    origins = [o.strip() for o in raw.split(",")]
    return [o for o in origins if o]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bank = BankLookup(os.getenv("BANK_DB", "wordbank.db"))
    app.state.wav_root = os.getenv("WAV_ROOT", "output/words")
    app.state.ogg_root = os.getenv("OGG_ROOT", "output/words_ogg")
    # Auth is env-gated (JWT_SECRET) so dev setups and existing deploys keep
    # working unauthenticated unless a secret is explicitly configured.
    if not os.getenv("JWT_SECRET"):
        log.warning("JWT_SECRET not set: /v1/speak, /v1/gemini/tts are unauthenticated")
    _warn_about_upstream_token(log)
    _warn_about_tts_style(log)
    yield
    app.state.bank.close()


app = FastAPI(title="TTS Server", lifespan=lifespan)


class _BodyLimit:
    """413 for a request body over MAX_BODY_BYTES, before the app sees any of
    it: FastAPI parses a form before require_auth runs. Pure ASGI."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        too_large = JSONResponse({"detail": f"request body exceeds {MAX_BODY_BYTES} bytes"}, 413)
        length = dict(scope["headers"]).get(b"content-length", b"0")
        if not length.isdigit() or int(length) > MAX_BODY_BYTES:
            return await too_large(scope, receive, send)
        body, more = b"", True
        while more:                     # a chunked body has no length: count it
            msg = await receive()
            if msg["type"] != "http.request":
                return                  # client gone
            body, more = body + msg.get("body", b""), msg.get("more_body", False)
            if len(body) > MAX_BODY_BYTES:
                return await too_large(scope, receive, send)
        first = [{"type": "http.request", "body": body, "more_body": False}]

        async def replay():
            return first.pop() if first else await receive()
        await self.app(scope, replay, send)


app.add_middleware(_BodyLimit)          # added first: CORS wraps it, so a 413 carries CORS headers

# CORS_ORIGINS env var: comma-separated allowlist (e.g.
# "https://app.example.com,https://admin.example.com"). Unset -> wildcard, so
# local dev and the current deployment keep working unchanged. NOTE: CORS is a
# browser-enforced control only — it stops a malicious *web page* from calling
# this API cross-origin, but does nothing against curl/scripts/servers, which
# ignore it entirely. It is not authentication and does not secure the API.
app.add_middleware(
    CORSMiddleware, allow_origins=_cors_origins(),
    allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"],
    # /v1/speak metadata; browsers hide non-safelisted headers from
    # cross-origin JS unless they're exposed.
    expose_headers=["X-Source", "X-Voice"],
)


@app.get("/health")
def health():
    bank = app.state.bank
    # Healthy iff the service can serve SOMETHING: the pre-generated bank or
    # /v1/speak get-or-generate (Gemini, or an upstream server in edge mode). A
    # deploy with no bank but a synth path is legitimate; only "none" is unhealthy.
    gemini = _gemini_configured()
    upstream = _upstream_url() is not None
    healthy = bank.available or gemini or upstream
    info = {
        "status": "ok",
        "healthy": healthy,
        "gemini": gemini,
        "upstream": upstream,
        "store": _speak_store(),
        "bank": bank.available,
        "bank_words": bank.count(),
    }
    return JSONResponse(content=info, status_code=200 if healthy else 503)


def _wav_abs(rel: str) -> str:
    """Resolve the relative manifest path under the configured WAV root."""
    return os.path.join(app.state.wav_root, rel)


def _ogg_abs(rel: str) -> str:
    """Mirror path under the OGG root, with a .ogg extension."""
    return os.path.join(app.state.ogg_root, os.path.splitext(rel)[0] + ".ogg")


def _file_info(path: Optional[str]) -> dict:
    exists = bool(path) and os.path.exists(path)
    return {"exists": exists, "bytes": os.path.getsize(path) if exists else None}


@app.get("/v1/languages")
def languages():
    """Language lists:

    - wordbank / phrasebank: data-driven — languages currently in each bank.
    - generate: manually-curated, verified list of languages to generate
      (wordbank/languages.py).
    """
    from wordbank.languages import SUPPORTED_LANGUAGES
    return {"wordbank": app.state.bank.languages(),
            "phrasebank": app.state.bank.phrase_languages(),
            "generate": SUPPORTED_LANGUAGES}


def _limit(limit: int) -> int:
    return max(1, min(limit, MAX_RESULTS))      # SQLite reads LIMIT -1 as "no limit"


@app.get("/v1/search")
def search(prefix: str, lang: str = DEFAULT_LANG, limit: int = 10,
           kind: Optional[str] = None, voice: Optional[str] = None):
    """Merged prefix search over the word + phrase banks. Each result carries a
    `type` flag ('word'|'phrase') and the `voice` it was generated with — one
    result per voice a text is banked in. `kind=word|phrase` restricts to one
    bank; `voice` to that voice (words and phrases by the same label, as
    /v1/speak matches them)."""
    return {"results": app.state.bank.search(prefix, _lang(lang), _limit(limit), kind,
                                             _bank_voice(voice))}


@app.get("/v1/bank")
def bank_suggest(prefix: str, lang: str = DEFAULT_LANG, limit: int = 10,
                 voice: Optional[str] = None):
    """Prefix autocomplete over the word bank only (index-backed range scan):
    distinct words, optionally only those banked in `voice`."""
    return {"words": app.state.bank.suggest(prefix, _lang(lang), _limit(limit),
                                            _bank_voice(voice))}


def _bank_word(word: str, lang: str, voice: Optional[str]):
    """The word's banked row: in `voice` when given, else the default voice's
    clip if banked, else the earliest banked; 404 when there is none."""
    row = app.state.bank.get(word, _lang(lang), voice=_bank_voice(voice),
                             prefer=_default_voice())
    if row is None:
        raise HTTPException(404, "not in bank")
    return row


def _serve_audio(rel: str, format: str, headers: Optional[dict] = None):
    """Serve the OGG copy (default — what production ships) or, for dev compare,
    the WAV master (?format=wav, only if WAV_ROOT is mounted)."""
    if format == "wav":
        wav = _wav_abs(rel)
        if not os.path.exists(wav):
            raise HTTPException(404, "wav master not available (production serves ogg)")
        return FileResponse(wav, media_type="audio/wav", headers=headers)
    ogg = _ogg_abs(rel)
    if not os.path.exists(ogg):
        raise HTTPException(404, "ogg copy not available")
    return FileResponse(ogg, media_type="audio/ogg", headers=headers)


@app.get("/v1/bank/{word}/info")
def bank_info(word: str, lang: str = DEFAULT_LANG, voice: Optional[str] = None):
    """Sizes/availability of the WAV master and OGG copy — for the compare UI.
    `voice` picks one of the voices the word is banked in (see bank_get)."""
    row = _bank_word(word, lang, voice)
    rel = row["file_path"]
    return {"word": row["word"], "type": "word", "voice": row["voice"],
            "wav": _file_info(_wav_abs(rel)), "ogg": _file_info(_ogg_abs(rel))}


@app.get("/v1/bank/{word}")
def bank_get(word: str, lang: str = DEFAULT_LANG, format: str = "ogg",
             voice: Optional[str] = None):
    """Serve a word's audio (OGG by default, WAV master in dev). A word can be
    banked in several voices: `voice` (case-insensitive label, "Kore",
    "en-gb-tutor-14") pins one, else the default GEMINI_TTS_VOICE's clip is
    served if banked, else the earliest banked."""
    return _serve_audio(_bank_word(word, lang, voice)["file_path"], format)


@app.get("/v1/phrase/info")
def phrase_info(text: str, lang: str = DEFAULT_LANG, voice: Optional[str] = None):
    """Sizes/availability for a phrase (optionally pin a voice: its label,
    "Kore"; the legacy "gemini-Kore" is accepted too)."""
    row = app.state.bank.get_phrase(text, _lang(lang), _bank_voice(voice))
    if row is None:
        raise HTTPException(404, "not in phrase bank")
    rel = row["file_path"]
    return {"text": row["text"], "type": "phrase", "voice": row["voice"],
            "wav": _file_info(_wav_abs(rel)), "ogg": _file_info(_ogg_abs(rel))}


@app.get("/v1/phrase")
def phrase_get(text: str, lang: str = DEFAULT_LANG, voice: Optional[str] = None,
               format: str = "ogg"):
    """Serve a phrase's audio (OGG by default, WAV master in dev); `voice` as
    for /v1/phrase/info."""
    row = app.state.bank.get_phrase(text, _lang(lang), _bank_voice(voice))
    if row is None:
        raise HTTPException(404, "not in phrase bank")
    return _serve_audio(row["file_path"], format)


def _bank_voice_headers(source: str, voice: Optional[str]) -> dict:
    """X-Source plus X-Voice (the voice label as /v1/search reports it). Labels
    of older local-voice clips are user-chosen; a non-ASCII one is omitted, since headers
    are latin-1 and a failed encode would 500 an otherwise good bank hit."""
    return {"X-Source": source, **({"X-Voice": voice} if voice and voice.isascii() else {})}


def _speak_bank_hit(text: str, lang: str, voice: Optional[str], eff_voice: str):
    """The banked recording /v1/speak should serve, or None. Single tokens may be
    in the word bank; anything may be in the phrase bank. Without a `voice`,
    the default voice (`eff_voice`) is preferred in both. A row whose OGG is
    missing on disk is passed over (a phrase may have a later row that holds
    it: one saved in other casing)."""
    if not any(c.isspace() for c in text):
        row = app.state.bank.get(text, lang, voice=voice, prefer=eff_voice)
        if row is not None and os.path.exists(_ogg_abs(row["file_path"])):
            row_voice = row["voice"] or voice_from_path(row["file_path"] or "")
            return _serve_audio(row["file_path"], "ogg",
                                headers=_bank_voice_headers("bank", row_voice))
    for pinned in ([eff_voice] if voice else [eff_voice, None]):
        for row in app.state.bank.phrases(text, lang, voice=pinned):
            if os.path.exists(_ogg_abs(row["file_path"])):
                return _serve_audio(row["file_path"], "ogg",
                                    headers=_bank_voice_headers("bank", row["voice"]))
    return None


# Single-flight for /v1/speak misses: (text, lang, voice) -> [lock, users].
# Concurrent requests for the same new clip queue behind the first, then find it
# banked, instead of each paying for its own generation. In-process, which is
# enough: the bank is one SQLite file, so a deployment is one server process.
_miss_locks: dict = {}

# With SPEAK_STORE=0 there is no saved clip for queued twins to find, so a
# store-less miss is coalesced instead: (key) -> the in-flight generation task,
# whose result every concurrent request for that clip shares.
_inflight: dict = {}


async def _coalesced(key: tuple, factory):
    task = _inflight.get(key)
    if task is None:
        task = asyncio.ensure_future(factory())
        _inflight[key] = task
        task.add_done_callback(lambda _t: _inflight.pop(key, None))
    # shield: one client disconnecting must not cancel the others' generation
    return await asyncio.shield(task)


@asynccontextmanager
async def _single_flight(key: tuple):
    entry = _miss_locks.setdefault(key, [asyncio.Lock(), 0])
    entry[1] += 1
    try:
        async with entry[0]:
            yield
    finally:
        entry[1] -= 1
        if entry[1] == 0:
            del _miss_locks[key]


def _in_executor(fn, *args):
    return asyncio.get_running_loop().run_in_executor(None, fn, *args)


async def _synthesize(text: str, voice: str, instruct: str, model: str,
                      style: Optional[str], server_style: bool = False) -> bytes:
    """Gemini synthesis off the event loop -> PCM; failures as HTTP errors with
    short details (the cause is logged). A style the model can't take is 400,
    or 500 when it is GEMINI_TTS_STYLE."""
    from wordbank.gemini_tts import synthesize_pcm  # lazy: genai
    try:
        return await _in_executor(lambda: synthesize_pcm(
            text, voice=voice, instruct=instruct, model=model, style=style))
    except GeminiStyleUnsupportedError as exc:
        if server_style:
            log.error("GEMINI_TTS_STYLE: %s", exc)
            raise HTTPException(500, "server misconfigured: GEMINI_TTS_STYLE") from exc
        raise HTTPException(400, str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(503, "google-genai not installed") from exc
    except GeminiNoAudioError as exc:           # our own message: the finish reason
        raise HTTPException(502, f"Gemini TTS failed: {exc}") from exc
    except Exception as exc:
        log.warning("Gemini TTS failed: %r", exc)
        raise HTTPException(502, "Gemini TTS failed") from exc


async def _encode_ogg(pcm: bytes) -> bytes:
    """PCM -> OGG in the bank profile, so saved clips match encode_wavs.py's."""
    from .audio import pcm16_to_ogg  # lazy: shells out to ffmpeg
    try:
        return await _in_executor(lambda: pcm16_to_ogg(
            pcm, bitrate=BANK_OGG_BITRATE, application=BANK_OGG_APPLICATION))
    except FileNotFoundError as exc:
        raise HTTPException(503, "ffmpeg not installed") from exc
    except subprocess.SubprocessError as exc:   # failed, or timed out
        log.error("OGG encoding failed: %s", getattr(exc, "stderr", None) or exc)
        raise HTTPException(500, "OGG encoding failed") from exc


@dataclass
class _Clip:
    ogg: bytes
    voice: str
    source: str                       # "gemini" | "upstream"
    pcm: Optional[bytes] = None       # Gemini only: the dev WAV master
    model: Optional[str] = None
    style: Optional[str] = None


async def _upstream_clip(url: str, text: str, lang: str, voice: Optional[str],
                         instruct: str, fallback_voice: str) -> _Clip:
    """A miss fetched from the upstream server, filed under the voice it
    reports, which must be a known voice: it names the saved clip's directory
    and is later shown by the frontend (an older upstream's "gemini-Kore" is
    "Kore")."""
    token = os.getenv("SPEAK_UPSTREAM_TOKEN")
    try:
        speech = await _in_executor(lambda: fetch_speech(
            url, token, text, lang, voice, instruct, timeout=SPEAK_UPSTREAM_TIMEOUT))
    except UpstreamError as exc:
        if exc.status >= 500:
            log.warning("upstream /v1/speak: %s (%r)", exc.detail, exc.__cause__)
        raise HTTPException(exc.status, exc.detail) from exc
    if not speech.voice:
        return _Clip(speech.ogg, fallback_voice, "upstream")
    label = canonical_voice(speech.voice.strip().removeprefix("gemini-"))
    if label is None:
        log.warning("upstream X-Voice %r is not a known voice", speech.voice)
        raise HTTPException(502, "upstream reported an unknown voice")
    return _Clip(speech.ogg, label, "upstream")


def _write(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def _save_phrase(text: str, lang: str, clip: _Clip) -> None:
    """Save a /v1/speak clip to the phrase bank, under its voice's directory
    in a `phrases/` subtree (phrase and word ids are separate sequences)."""
    from wordbank.manifest import DONE, Manifest
    from wordbank.paths import path_under, safe_filename
    clip_dir = os.path.join(lang, f"gemini-{clip.voice}", "phrases")
    try:   # last line of defence, before the row exists
        for root in (app.state.ogg_root, app.state.wav_root):
            path_under(root, clip_dir)
    except ValueError as exc:
        raise HTTPException(500, "clip path outside the bank") from exc
    db = os.getenv("BANK_DB", "wordbank.db")
    os.makedirs(os.path.dirname(db) or ".", exist_ok=True)   # bank-less deploy: first write creates it
    m = Manifest(db, table="phrases", text_col="text")
    try:
        m.add_words([text], lang, voice=clip.voice)
        row = m.get_entry(text, lang, clip.voice)
        try:
            rel = os.path.join(clip_dir, str(row["id"] // 1000),
                               f"{row['id']}_{safe_filename(text)}.wav")
            _write(_ogg_abs(rel), clip.ogg)
            if clip.pcm is not None and os.path.isdir(app.state.wav_root):   # dev-only WAV master
                from wordbank.gemini_tts import pcm_to_wav_bytes
                try:
                    _write(_wav_abs(rel), pcm_to_wav_bytes(clip.pcm))
                except OSError:
                    pass
            m.set_status(row["id"], DONE, file_path=rel, voice=clip.voice, model=clip.model,
                         style=clip.style,
                         billed_chars=len(text) if clip.source == "gemini" else None)
        except Exception:
            m.discard_unfinished(row["id"])     # no pending row left behind
            raise
    finally:
        m.close()
    app.state.bank.close()   # drop the memoized read conn so the new row is visible


def _clip_response(clip: _Clip) -> Response:
    return Response(content=clip.ogg, media_type="audio/ogg",
                    headers=_bank_voice_headers(clip.source, clip.voice))


@app.post("/v1/speak", dependencies=[Depends(require_auth)])
async def speak(
    text: str = Form(...),
    lang: Annotated[str, Form()] = DEFAULT_LANG,
    voice: Annotated[Optional[str], Form()] = None,
    instruct: Annotated[Optional[str], Form()] = None,
):
    """Get-or-generate: serve `text` from the word or phrase bank, else
    generate it (Gemini, or the upstream in edge mode), save it to the phrase
    bank (not with SPEAK_STORE=0) and serve it. `voice` pins the recording;
    omitted, any banked voice serves and a miss uses GEMINI_TTS_VOICE.
    `instruct` makes a one-off: no lookup, no save. Headers: X-Source
    (bank|gemini|upstream), X-Voice. See README for the full contract."""
    text = text.strip()
    if not text:
        raise HTTPException(400, "text must not be empty")
    if len(text) > MAX_TEXT_LEN:
        raise HTTPException(400, f"text exceeds {MAX_TEXT_LEN} characters")
    lang = _lang(lang)
    voice = _gemini_voice(voice)
    instruct = _direction(instruct, "instruct")
    eff_voice = voice or _default_voice()
    upstream = _upstream_url()

    async def generate() -> _Clip:
        if upstream:
            return await _upstream_clip(upstream, text, lang, voice, instruct, eff_voice)
        _check_voice(eff_voice, GEMINI_TTS_MODEL, server_default=not voice)
        style = None if instruct else _tts_style()   # a one-off's instruct replaces it
        pcm = await _synthesize(text, eff_voice, instruct, GEMINI_TTS_MODEL, style,
                                server_style=True)
        return _Clip(await _encode_ogg(pcm), eff_voice, "gemini", pcm, GEMINI_TTS_MODEL, style)

    if instruct:
        return _clip_response(await generate())
    if (hit := _speak_bank_hit(text, lang, voice, eff_voice)) is not None:
        return hit
    key = (text.casefold(), lang, eff_voice)
    if not _speak_store():
        return _clip_response(await _coalesced(key, generate))
    async with _single_flight(key):
        if (hit := _speak_bank_hit(text, lang, voice, eff_voice)) is not None:
            return hit                  # a queued twin saved it
        clip = await generate()
        try:
            await _in_executor(_save_phrase, text, lang, clip)
        except Exception:                       # the clip is paid for: serve it anyway
            log.exception("could not save the /v1/speak clip for %r", text[:80])
    return _clip_response(clip)


# --- Voice studio: preview Gemini prebuilt voices ---

@app.post("/v1/gemini/tts", dependencies=[Depends(require_auth)])
async def gemini_tts(
    text: str = Form(...),
    voice: Annotated[str, Form()] = GEMINI_TTS_VOICE,
    instruct: Annotated[Optional[str], Form()] = None,
    model: Annotated[str, Form()] = GEMINI_TTS_MODEL,
    style: Annotated[Optional[str], Form()] = None,
):
    """Preview a phrase in a Gemini voice -> WAV; nothing saved. `voice`,
    `instruct` as for /v1/speak; `style` (3.8 models only) as speech metadata.
    GEMINI_TTS_STYLE does not apply."""
    if len(text) > MAX_TEXT_LEN:
        raise HTTPException(400, f"text exceeds {MAX_TEXT_LEN} characters")
    if model not in TTS_MODELS and model != GEMINI_TTS_MODEL:
        raise HTTPException(400, f"unknown model {model!r} (one of: {', '.join(TTS_MODELS)})")
    voice = _gemini_voice(voice) or GEMINI_TTS_VOICE
    _check_voice(voice, model)
    from wordbank.gemini_tts import pcm_to_wav_bytes
    pcm = await _synthesize(text, voice, _direction(instruct, "instruct"), model,
                            _direction(style, "style") or None)
    return Response(content=pcm_to_wav_bytes(pcm), media_type="audio/wav")


@app.get("/v1/gemini/voices")
def gemini_voices():
    """The 30 classic Gemini TTS voices (static list, name + style + gender),
    which /v1/speak and /v1/gemini/tts accept on every model. They aren't
    language-bound — Gemini auto-detects language. The curated per-language
    voices are listed by /v1/gemini/languages."""
    from wordbank.gemini_tts import VOICES
    return {"voices": [{"name": n, "style": s, "gender": g} for n, s, g in VOICES]}


@app.get("/v1/gemini/languages")
def gemini_languages():
    """The curated Gemini voices per language (wordbank/gemini_voices.json):
    each bank `lang` code with one female and one male voice and their voice
    library metadata. A voice's `name` is what /v1/speak and /v1/gemini/tts
    take as `voice`, and the label its words and phrases are banked under
    (filed in `<lang>/gemini-<name>/`).
    `classic` marks the language-agnostic classic voices (en-us uses two);
    the others are locale voices, which need a 3.8 model (`locale_voice_models`)."""
    from wordbank.gemini_tts import locale_voice_models
    from wordbank.languages import VOICE_LANGUAGES
    return {
        "languages": [{
            "code": lang["code"], "name": lang["name"],
            "voices": [{"name": canonical_voice(v["id"]), "classic": is_classic_voice(v["id"]),
                        **{k: x for k, x in v.items() if k != "id"}}
                       for v in lang["voices"]],
        } for lang in VOICE_LANGUAGES],
        "locale_voice_models": locale_voice_models(),
    }


# Optionally serve the static frontend from this server (same origin → relative
# API calls just work). Mounted last so API routes win. Enable with SERVE_FRONTEND.
if _truthy(os.getenv("SERVE_FRONTEND")) and os.path.isdir(FRONTEND_DIR):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
