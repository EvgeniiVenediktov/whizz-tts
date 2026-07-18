import asyncio
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Annotated, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from wordbank.gemini_tts import DEFAULT_MODEL as GEMINI_TTS_DEFAULT_MODEL
from wordbank.lookup import BankLookup

# NOTE: src.model / src.audio (torch, omnivoice, numpy, soundfile) are imported
# LAZILY inside the synthesis handler — so a bank-only build with MODEL_ENABLED
# off needs none of the ML stack installed.

MODEL_ID = os.getenv("MODEL_ID", "k2-fsa/OmniVoice")
MAX_TEXT_LEN = int(os.getenv("MAX_TEXT_LEN", "2000"))
MAX_REF_BYTES = int(os.getenv("MAX_REF_BYTES", str(10 * 1024 * 1024)))  # 10 MB
FRONTEND_DIR = os.getenv("FRONTEND_DIR", "frontend")
DEFAULT_LANG = os.getenv("DEFAULT_LANG", "en-us")
GEMINI_TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", GEMINI_TTS_DEFAULT_MODEL)
GEMINI_TTS_VOICE = os.getenv("GEMINI_TTS_VOICE", "Kore")


def _truthy(val: Optional[str]) -> bool:
    return (val or "").lower() not in ("", "0", "false", "no", "off")


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
)


@app.get("/health")
def health():
    bank = app.state.bank
    # Healthy iff the service can serve SOMETHING: the pre-generated bank, or
    # live synthesis (MODEL_ENABLED). A model-only deploy with no bank is a
    # legitimate config and must stay healthy; only "neither" is unhealthy.
    healthy = bank.available or app.state.model_enabled
    info = {
        "status": "ok",
        "healthy": healthy,
        "model": MODEL_ID,
        "model_enabled": app.state.model_enabled,
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


def _serve_audio(rel: str, format: str):
    """Serve the OGG copy (default — what production ships) or, for dev compare,
    the WAV master (?format=wav, only if WAV_ROOT is mounted)."""
    if format == "wav":
        wav = _wav_abs(rel)
        if not os.path.exists(wav):
            raise HTTPException(404, "wav master not available (production serves ogg)")
        return FileResponse(wav, media_type="audio/wav")
    ogg = _ogg_abs(rel)
    if not os.path.exists(ogg):
        raise HTTPException(404, "ogg copy not available")
    return FileResponse(ogg, media_type="audio/ogg")


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


@app.post("/v1/tts")
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

    from .audio import pcm_to_ogg  # lazy: needs numpy/soundfile/ffmpeg
    return Response(content=pcm_to_ogg(audio), media_type="audio/ogg")


# --- Voice studio: author Gemini references, save + reuse them for cloning ---

@app.post("/v1/gemini/tts")
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


@app.post("/v1/voices")
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
