import asyncio
import logging
import os
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Annotated, Optional

import jwt
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from .auth import require_auth
from .upstream import UpstreamError, fetch_speech
from wordbank.constants import BANK_OGG_APPLICATION, BANK_OGG_BITRATE
from wordbank.gemini_tts import DEFAULT_MODEL as GEMINI_TTS_DEFAULT_MODEL, canonical_voice
from wordbank.lookup import BankLookup
from wordbank.paths import voice_from_path

# NOTE: src.model (torch, omnivoice) and src.audio (ffmpeg) are imported LAZILY
# inside the synthesis handlers — so a bank-only build with MODEL_ENABLED off
# needs none of the ML stack installed.

MODEL_ID = os.getenv("MODEL_ID", "k2-fsa/OmniVoice")
MAX_TEXT_LEN = int(os.getenv("MAX_TEXT_LEN", "2000"))
MAX_REF_BYTES = int(os.getenv("MAX_REF_BYTES", str(10 * 1024 * 1024)))  # 10 MB
FRONTEND_DIR = os.getenv("FRONTEND_DIR", "frontend")
DEFAULT_LANG = os.getenv("DEFAULT_LANG", "en-us")
GEMINI_TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", GEMINI_TTS_DEFAULT_MODEL)
GEMINI_TTS_VOICE = os.getenv("GEMINI_TTS_VOICE", "Kore")
SPEAK_UPSTREAM_TIMEOUT = float(os.getenv("SPEAK_UPSTREAM_TIMEOUT", "60"))


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
    """Canonical prebuilt-voice name for a request field, None when omitted.
    An unknown name is a 400 here rather than a 502 from a wasted Gemini call."""
    if not voice or not voice.strip():
        return None
    canon = canonical_voice(voice)
    if canon is None:
        raise HTTPException(400, f"unknown Gemini voice {voice!r} (see /v1/gemini/voices)")
    return canon


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
    app.state.voices_root = os.getenv("VOICES_ROOT", "output/voices")
    app.state.model_enabled = _truthy(os.getenv("MODEL_ENABLED"))
    if app.state.model_enabled and _truthy(os.getenv("PRELOAD_MODEL")):
        from .model import load_model
        load_model(MODEL_ID)
    # Auth is env-gated (JWT_SECRET) so dev setups and existing deploys keep
    # working unauthenticated unless a secret is explicitly configured.
    log = logging.getLogger("uvicorn.error")
    if not os.getenv("JWT_SECRET"):
        log.warning("JWT_SECRET not set: /v1/tts, /v1/speak, /v1/gemini/tts, /v1/voices are unauthenticated")
    _warn_about_upstream_token(log)
    yield
    app.state.bank.close()


app = FastAPI(title="TTS Server", lifespan=lifespan)

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
    # Healthy iff the service can serve SOMETHING: the pre-generated bank, live
    # synthesis (MODEL_ENABLED), or /v1/speak get-or-generate (Gemini, or an
    # upstream server in edge mode). A deploy with no bank but any synth path
    # is legitimate; only "none" is unhealthy.
    gemini = _gemini_configured()
    upstream = _upstream_url() is not None
    healthy = bank.available or app.state.model_enabled or gemini or upstream
    info = {
        "status": "ok",
        "healthy": healthy,
        "model": MODEL_ID,
        "model_enabled": app.state.model_enabled,
        "gemini": gemini,
        "upstream": upstream,
        "bank": bank.available,
        "bank_path": bank.path,
        "bank_words": bank.count(),
    }
    if app.state.model_enabled:
        try:
            from .model import get_device, is_loaded
            info["device"] = get_device()
            info["model_loaded"] = is_loaded()
        except Exception:
            info["model_loaded"] = False
    return JSONResponse(content=info, status_code=200 if healthy else 503)


def _media_type(path: str) -> str:
    return "audio/ogg" if path.endswith(".ogg") else "audio/wav"


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
    - generate: manually-curated, verified list the model can generate
      (wordbank/languages.py).
    """
    from wordbank.languages import SUPPORTED_LANGUAGES
    return {"wordbank": app.state.bank.languages(),
            "phrasebank": app.state.bank.phrase_languages(),
            "generate": SUPPORTED_LANGUAGES}


@app.get("/v1/search")
def search(prefix: str, lang: str = DEFAULT_LANG, limit: int = 10,
           kind: Optional[str] = None):
    """Merged prefix search over the word + phrase banks. Each result carries a
    `type` flag ('word'|'phrase') and the `voice` it was generated with.
    `kind=word|phrase` restricts to one bank."""
    return {"results": app.state.bank.search(prefix, lang, min(limit, 50), kind)}


@app.get("/v1/bank")
def bank_suggest(prefix: str, lang: str = DEFAULT_LANG, limit: int = 10):
    """Prefix autocomplete over the word bank only (index-backed range scan)."""
    return {"words": app.state.bank.suggest(prefix, lang, min(limit, 50))}


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
def bank_info(word: str, lang: str = DEFAULT_LANG):
    """Sizes/availability of the WAV master and OGG copy — for the compare UI."""
    row = app.state.bank.get(word, lang)
    if row is None:
        raise HTTPException(404, "not in bank")
    rel = row["file_path"]
    return {"word": row["word"], "type": "word", "voice": row["voice"],
            "wav": _file_info(_wav_abs(rel)), "ogg": _file_info(_ogg_abs(rel))}


@app.get("/v1/bank/{word}")
def bank_get(word: str, lang: str = DEFAULT_LANG, format: str = "ogg"):
    """Serve a word's audio (OGG by default, WAV master in dev)."""
    row = app.state.bank.get(word, lang)
    if row is None:
        raise HTTPException(404, "not in bank")
    return _serve_audio(row["file_path"], format)


@app.get("/v1/phrase/info")
def phrase_info(text: str, lang: str = DEFAULT_LANG, voice: Optional[str] = None):
    """Sizes/availability for a phrase (optionally pin a voice)."""
    row = app.state.bank.get_phrase(text, lang, voice)
    if row is None:
        raise HTTPException(404, "not in phrase bank")
    rel = row["file_path"]
    return {"text": row["text"], "type": "phrase", "voice": row["voice"],
            "seed": row["seed"], "ref_audio": row["ref_audio"],
            "ref_text": row["ref_text"],
            "wav": _file_info(_wav_abs(rel)), "ogg": _file_info(_ogg_abs(rel))}


@app.get("/v1/phrase")
def phrase_get(text: str, lang: str = DEFAULT_LANG, voice: Optional[str] = None,
               format: str = "ogg"):
    """Serve a phrase's audio (OGG by default, WAV master in dev)."""
    row = app.state.bank.get_phrase(text, lang, voice)
    if row is None:
        raise HTTPException(404, "not in phrase bank")
    return _serve_audio(row["file_path"], format)


def _bank_voice_headers(source: str, voice: Optional[str]) -> dict:
    """X-Source plus X-Voice (the voice label as /v1/search reports it). Labels
    of local voices are user-chosen; a non-ASCII one is omitted, since headers
    are latin-1 and a failed encode would 500 an otherwise good bank hit."""
    return {"X-Source": source, **({"X-Voice": voice} if voice and voice.isascii() else {})}


def _speak_bank_hit(text: str, lang: str, voice: Optional[str], bank_voice: str):
    """The banked recording /v1/speak should serve, or None. Single tokens may be
    in the word bank; anything may be in the phrase bank. A manifest row whose
    OGG is missing on disk is a MISS."""
    if not any(c.isspace() for c in text):
        row = app.state.bank.get(text, lang, voice=voice)
        if row is not None and os.path.exists(_ogg_abs(row["file_path"])):
            row_voice = row["voice"] or voice_from_path(row["file_path"] or "")
            return _serve_audio(row["file_path"], "ogg",
                                headers=_bank_voice_headers("bank", row_voice))
    for pinned in ([bank_voice] if voice else [bank_voice, None]):
        row = app.state.bank.get_phrase(text, lang, voice=pinned)
        if row is not None and os.path.exists(_ogg_abs(row["file_path"])):
            return _serve_audio(row["file_path"], "ogg",
                                headers=_bank_voice_headers("bank", row["voice"]))
    return None


# Single-flight for /v1/speak misses: (text, lang, bank voice) -> [lock, users].
# Concurrent requests for the same new clip queue behind the first, then find it
# banked, instead of each paying for its own generation. In-process, which is
# enough: the bank is one SQLite file, so a deployment is one server process.
_miss_locks: dict = {}


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


async def _gemini_clip(text: str, voice: str, instruct: str) -> tuple:
    """Synthesize with Gemini -> (OGG in the bank profile, raw PCM)."""
    def _gen():
        from wordbank.gemini_tts import synthesize_pcm  # lazy: genai
        return synthesize_pcm(text, voice=voice, instruct=instruct, model=GEMINI_TTS_MODEL)

    try:
        pcm = await asyncio.get_running_loop().run_in_executor(None, _gen)
    except (ImportError, ModuleNotFoundError) as exc:
        raise HTTPException(503, f"google-genai not installed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(502, f"Gemini TTS failed: {exc}") from exc

    # Encoded in the bank's own OGG profile, so a clip saved here matches the
    # batch-encoded ones (scripts/encode_wavs.py uses the same constants).
    from .audio import pcm16_to_ogg  # lazy: shells out to ffmpeg
    try:
        ogg = pcm16_to_ogg(pcm, bitrate=BANK_OGG_BITRATE, application=BANK_OGG_APPLICATION)
    except FileNotFoundError as exc:
        raise HTTPException(503, f"ffmpeg not installed: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        raise HTTPException(500, f"OGG encoding failed: {exc.stderr.decode(errors='replace')}") from exc
    return ogg, pcm


async def _upstream_clip(url: str, text: str, lang: str, voice: Optional[str],
                         instruct: str, fallback_voice: str) -> tuple:
    """Fetch a miss from the upstream server -> (OGG, bank voice label). The
    voice is forwarded as requested (None stays None), so upstream applies its
    own default/any-banked rules; the clip is filed under the voice upstream
    reports. A word-bank hit there reports a bare Gemini name ("Kore"), which
    is filed as the phrase voice "gemini-Kore" here."""
    token = os.getenv("SPEAK_UPSTREAM_TOKEN")
    try:
        speech = await asyncio.get_running_loop().run_in_executor(
            None, lambda: fetch_speech(url, token, text, lang, voice, instruct,
                                       timeout=SPEAK_UPSTREAM_TIMEOUT))
    except UpstreamError as exc:
        raise HTTPException(exc.status, exc.detail) from exc
    label = speech.voice
    if label and canonical_voice(label):
        label = f"gemini-{canonical_voice(label)}"
    return speech.ogg, label or fallback_voice


def _save_phrase(text: str, lang: str, bank_voice: str, ogg: bytes,
                 pcm: Optional[bytes] = None, billed_chars: Optional[int] = None) -> None:
    """Persist a /v1/speak clip into the phrase bank (mirrors the save_voice
    writer below). Phrases get their own `phrases/` subtree: phrase and word ids
    are separate sequences, so `<lang>/<voice>/<n>/<id>_<slug>` alone could
    collide with a word clip of the same id, voice, and slug."""
    from wordbank.manifest import DONE, Manifest
    from wordbank.paths import safe_filename
    db = os.getenv("BANK_DB", "wordbank.db")
    os.makedirs(os.path.dirname(db) or ".", exist_ok=True)   # bank-less deploy: first write creates it
    m = Manifest(db, table="phrases", text_col="text")
    try:
        m.add_words([text], lang, voice=bank_voice)
        row = m.get_entry(text, lang, bank_voice)
        rel = os.path.join(lang, bank_voice, "phrases", str(row["id"] // 1000),
                           f"{row['id']}_{safe_filename(text)}.wav")
        ogg_path = _ogg_abs(rel)
        os.makedirs(os.path.dirname(ogg_path), exist_ok=True)
        with open(ogg_path, "wb") as f:
            f.write(ogg)
        if pcm is not None and os.path.isdir(app.state.wav_root):   # dev-only WAV master
            try:
                from wordbank.gemini_tts import pcm_to_wav_bytes
                wav_path = _wav_abs(rel)
                os.makedirs(os.path.dirname(wav_path), exist_ok=True)
                with open(wav_path, "wb") as f:
                    f.write(pcm_to_wav_bytes(pcm))
            except OSError:
                pass
        m.set_status(row["id"], DONE, file_path=rel, billed_chars=billed_chars, voice=bank_voice)
    finally:
        m.close()
    app.state.bank.close()   # drop the memoized read conn so the new row is visible


@app.post("/v1/speak", dependencies=[Depends(require_auth)])
async def speak(
    text: str = Form(...),
    lang: Annotated[str, Form()] = DEFAULT_LANG,
    voice: Annotated[Optional[str], Form()] = None,
    instruct: Annotated[Optional[str], Form()] = None,
):
    """Get-or-generate: serve `text` from the bank (word or phrase) if it's
    already there, otherwise generate it and persist it into the phrase bank so
    subsequent requests are served from disk. Along with `/v1/voices`, this is
    the only other route that writes the bank.

    A miss is synthesized with Gemini, or — in edge mode (SPEAK_UPSTREAM_URL
    set) — fetched from the upstream server's /v1/speak, so an edge needs no
    Gemini key. Concurrent misses for the same clip are generated once.

    `voice` is a Gemini prebuilt voice name (case-insensitive). When given, only
    a recording in that voice is served — words match the bare name ("Kore"),
    phrases the bank voice `gemini-<voice>` — and a miss is generated in it.
    When omitted, any banked recording serves (the default GEMINI_TTS_VOICE
    first) and a miss is generated in the default voice.

    `instruct` (style direction) makes the request a one-off: the bank has no
    key for style, so it is neither looked up nor saved.

    Response headers: `X-Source: bank|gemini|upstream`, `X-Voice: <voice label>`.
    """
    text = text.strip()
    if not text:
        raise HTTPException(400, "text must not be empty")
    if len(text) > MAX_TEXT_LEN:
        raise HTTPException(400, f"text exceeds {MAX_TEXT_LEN} characters")
    voice = _gemini_voice(voice)
    instruct = (instruct or "").strip()
    eff_voice = voice or canonical_voice(GEMINI_TTS_VOICE) or GEMINI_TTS_VOICE
    bank_voice = f"gemini-{eff_voice}"
    upstream = _upstream_url()

    async def _generate():
        if upstream:
            ogg, label = await _upstream_clip(upstream, text, lang, voice, instruct, bank_voice)
            return ogg, label, "upstream", None, None
        ogg, pcm = await _gemini_clip(text, eff_voice, instruct)
        return ogg, bank_voice, "gemini", pcm, len(text)

    if instruct:
        ogg, label, source, _, _ = await _generate()
        return Response(content=ogg, media_type="audio/ogg",
                        headers=_bank_voice_headers(source, label))

    hit = _speak_bank_hit(text, lang, voice, bank_voice)
    if hit is not None:
        return hit
    async with _single_flight((text.lower(), lang, bank_voice)):
        hit = _speak_bank_hit(text, lang, voice, bank_voice)   # a queued twin may have saved it
        if hit is not None:
            return hit
        ogg, label, source, pcm, billed = await _generate()
        _save_phrase(text, lang, label, ogg, pcm, billed)
    return Response(content=ogg, media_type="audio/ogg",
                    headers=_bank_voice_headers(source, label))


@app.post("/v1/tts", dependencies=[Depends(require_auth)])
async def tts(
    text: str = Form(...),
    ref_text: Annotated[Optional[str], Form()] = None,
    ref_audio: Annotated[Optional[UploadFile], File()] = None,
    voice: Annotated[Optional[str], Form()] = None,
    lang: Annotated[str, Form()] = DEFAULT_LANG,
    instruct: Annotated[Optional[str], Form()] = None,
    guidance_scale: Annotated[Optional[float], Form()] = None,
    seed: Annotated[Optional[int], Form()] = None,
):
    """Live synthesis (requires MODEL_ENABLED + the model build).

    Modes: ref_audio upload[+ref_text] → cloning; `voice` (a saved reference
    name) → clone from the stored clip; instruct → voice design; none → auto.
    """
    if not app.state.model_enabled:
        raise HTTPException(
            503, "live synthesis disabled (MODEL_ENABLED off); this build serves the bank only"
        )

    if not ref_text:
        ref_text = None
    if not instruct:
        instruct = None
    if len(text) > MAX_TEXT_LEN:
        raise HTTPException(400, f"text exceeds {MAX_TEXT_LEN} characters")

    ref_path: Optional[str] = None
    cleanup_ref = False
    if ref_audio is not None:                       # uploaded clip (one-off)
        raw = await ref_audio.read(MAX_REF_BYTES + 1)
        if len(raw) > MAX_REF_BYTES:
            raise HTTPException(413, "ref_audio exceeds size limit")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_file:
            ref_path = ref_file.name
            ref_file.write(raw)
        cleanup_ref = True
    elif voice:                                     # saved reference by name
        from wordbank.voices import resolve_voice_path
        row = app.state.bank.get_voice(voice, lang)
        if row is None:
            raise HTTPException(404, f"no saved voice {voice!r} for lang {lang!r}")
        ref_path = resolve_voice_path(app.state.voices_root, row["ref_audio"])
        if not os.path.exists(ref_path):
            raise HTTPException(404, f"reference clip missing on disk: {row['ref_audio']}")
        if ref_text is None:
            ref_text = row["ref_text"]

    def _synthesize():
        from .model import ensure_loaded, generate  # lazy: needs torch/omnivoice
        ensure_loaded(MODEL_ID)
        return generate(text, ref_path, ref_text, instruct, guidance_scale, seed)

    try:
        loop = asyncio.get_running_loop()
        audio = await loop.run_in_executor(None, _synthesize)
    except (ImportError, ModuleNotFoundError) as exc:
        raise HTTPException(503, f"synthesis model not available in this build: {exc}") from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    finally:
        if cleanup_ref and ref_path is not None:
            os.unlink(ref_path)

    from .audio import pcm_to_ogg  # lazy: shells out to ffmpeg
    return Response(content=pcm_to_ogg(audio), media_type="audio/ogg")


# --- Voice studio: author Gemini references, save + reuse them for cloning ---

@app.post("/v1/gemini/tts", dependencies=[Depends(require_auth)])
async def gemini_tts(
    text: str = Form(...),
    voice: Annotated[str, Form()] = GEMINI_TTS_VOICE,
    instruct: Annotated[Optional[str], Form()] = None,
    model: Annotated[str, Form()] = GEMINI_TTS_MODEL,
):
    """Generate a phrase with Gemini (a prebuilt voice) → WAV, for previewing
    before saving as a clone reference. Needs GEMINI_API_KEY on the server."""
    if len(text) > MAX_TEXT_LEN:
        raise HTTPException(400, f"text exceeds {MAX_TEXT_LEN} characters")
    voice = _gemini_voice(voice) or GEMINI_TTS_VOICE

    def _gen():
        from wordbank.gemini_tts import pcm_to_wav_bytes, synthesize_pcm  # lazy: genai
        pcm = synthesize_pcm(text, voice=voice, instruct=instruct or "", model=model)
        return pcm_to_wav_bytes(pcm)

    try:
        loop = asyncio.get_running_loop()
        wav = await loop.run_in_executor(None, _gen)
    except (ImportError, ModuleNotFoundError) as exc:
        raise HTTPException(503, f"google-genai not installed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(502, f"Gemini TTS failed: {exc}") from exc
    return Response(content=wav, media_type="audio/wav")


@app.get("/v1/gemini/voices")
def gemini_voices():
    """The prebuilt Gemini TTS voices (static list, name + style) for the studio
    dropdown. Voices aren't language-bound — Gemini auto-detects language."""
    from wordbank.gemini_tts import VOICES
    return {"voices": [{"name": n, "style": s, "gender": g} for n, s, g in VOICES]}


@app.get("/v1/voices")
def list_voices(lang: Optional[str] = None):
    """Saved clone references (optionally one language) — for the picker UI."""
    return {"voices": app.state.bank.voices(lang)}


@app.get("/v1/voices/{lang}/{name}/audio")
def voice_audio(lang: str, name: str):
    """Serve a saved reference clip so the UI can replay it."""
    row = app.state.bank.get_voice(name, lang)
    if row is None:
        raise HTTPException(404, "no such voice")
    from wordbank.voices import resolve_voice_path
    path = resolve_voice_path(app.state.voices_root, row["ref_audio"])
    if not os.path.exists(path):
        raise HTTPException(404, "reference clip missing on disk")
    return FileResponse(path, media_type=_media_type(path))


@app.post("/v1/voices", dependencies=[Depends(require_auth)])
async def save_voice(
    name: str = Form(...),
    ref_text: str = Form(...),
    audio: UploadFile = File(...),
    lang: Annotated[str, Form()] = DEFAULT_LANG,
    instruct: Annotated[Optional[str], Form()] = None,
    source: Annotated[str, Form()] = "gemini",
):
    """Save an audio clip as a reusable OmniVoice clone reference (writes the WAV
    under VOICES_ROOT and registers it in the manifest's `voices` table)."""
    raw = await audio.read(MAX_REF_BYTES + 1)
    if len(raw) > MAX_REF_BYTES:
        raise HTTPException(413, "audio exceeds size limit")
    if raw[:4] != b"RIFF":
        raise HTTPException(400, "audio must be a WAV file")

    from wordbank.manifest import Manifest
    from wordbank.voices import save_voice_clip
    m = Manifest(os.getenv("BANK_DB", "wordbank.db"))
    try:
        rel = save_voice_clip(m, app.state.voices_root, name, lang, raw,
                              ref_text, instruct=(instruct or None), source=source)
    finally:
        m.close()
    app.state.bank.close()   # drop the memoized read conn so the new row is visible
    return {"name": name, "lang": lang, "ref_audio": rel, "ref_text": ref_text,
            "instruct": instruct or None, "source": source}


# Optionally serve the static frontend from this server (same origin → relative
# API calls just work). Mounted last so API routes win. Enable with SERVE_FRONTEND.
if _truthy(os.getenv("SERVE_FRONTEND")) and os.path.isdir(FRONTEND_DIR):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
