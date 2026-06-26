import asyncio
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Annotated, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from wordbank.lookup import BankLookup

# NOTE: src.model / src.audio (torch, omnivoice, numpy, soundfile) are imported
# LAZILY inside the synthesis handler — so a bank-only build with MODEL_ENABLED
# off needs none of the ML stack installed.

MODEL_ID = os.getenv("MODEL_ID", "k2-fsa/OmniVoice")
MAX_TEXT_LEN = int(os.getenv("MAX_TEXT_LEN", "2000"))
MAX_REF_BYTES = int(os.getenv("MAX_REF_BYTES", str(10 * 1024 * 1024)))  # 10 MB
FRONTEND_DIR = os.getenv("FRONTEND_DIR", "frontend")
DEFAULT_LANG = os.getenv("DEFAULT_LANG", "en-us")


def _truthy(val: Optional[str]) -> bool:
    return (val or "").lower() not in ("", "0", "false", "no", "off")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bank = BankLookup(os.getenv("BANK_DB", "wordbank.db"))
    app.state.wav_root = os.getenv("WAV_ROOT", "output/words")
    app.state.ogg_root = os.getenv("OGG_ROOT", "output/words_ogg")
    app.state.model_enabled = _truthy(os.getenv("MODEL_ENABLED"))
    if app.state.model_enabled and _truthy(os.getenv("PRELOAD_MODEL")):
        from .model import load_model
        load_model(MODEL_ID)
    yield
    app.state.bank.close()


app = FastAPI(title="TTS Server", lifespan=lifespan)

# CORS open for convenience; restrict allow_origins in production.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
def health():
    bank = app.state.bank
    info = {
        "status": "ok",
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
    return info


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
    """Two language lists:

    - wordbank: data-driven — languages currently in the bank (lang + ready count).
    - generate: manually-curated, verified list the model can generate
      (wordbank/languages.py).
    """
    from wordbank.languages import SUPPORTED_LANGUAGES
    return {"wordbank": app.state.bank.languages(), "generate": SUPPORTED_LANGUAGES}


@app.get("/v1/bank")
def bank_suggest(prefix: str, lang: str = DEFAULT_LANG, limit: int = 10):
    """Prefix autocomplete over the bank (index-backed range scan)."""
    return {"words": app.state.bank.suggest(prefix, lang, min(limit, 50))}


@app.get("/v1/bank/{word}/info")
def bank_info(word: str, lang: str = DEFAULT_LANG):
    """Sizes/availability of the WAV master and OGG copy — for the compare UI."""
    row = app.state.bank.get(word, lang)
    if row is None:
        raise HTTPException(404, "not in bank")
    rel = row["file_path"]
    return {"word": row["word"], "wav": _file_info(_wav_abs(rel)),
            "ogg": _file_info(_ogg_abs(rel))}


@app.get("/v1/bank/{word}")
def bank_get(word: str, lang: str = DEFAULT_LANG, format: str = "ogg"):
    """Serve the OGG copy (default — what production ships) or, for dev compare,
    the WAV master (?format=wav, only if WAV_ROOT is mounted)."""
    row = app.state.bank.get(word, lang)
    if row is None:
        raise HTTPException(404, "not in bank")
    rel = row["file_path"]
    if format == "wav":
        wav = _wav_abs(rel)
        if not os.path.exists(wav):
            raise HTTPException(404, "wav master not available (production serves ogg)")
        return FileResponse(wav, media_type="audio/wav")
    ogg = _ogg_abs(rel)
    if not os.path.exists(ogg):
        raise HTTPException(404, "ogg copy not available")
    return FileResponse(ogg, media_type="audio/ogg")


@app.post("/v1/tts")
async def tts(
    text: str = Form(...),
    ref_text: Annotated[Optional[str], Form()] = None,
    ref_audio: Annotated[Optional[UploadFile], File()] = None,
    instruct: Annotated[Optional[str], Form()] = None,
    guidance_scale: Annotated[Optional[float], Form()] = None,
    seed: Annotated[Optional[int], Form()] = None,
):
    """Live synthesis (requires MODEL_ENABLED + the model build).

    Modes: ref_audio[+ref_text] → cloning; instruct → voice design; none → auto.
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
    if ref_audio is not None:
        raw = await ref_audio.read(MAX_REF_BYTES + 1)
        if len(raw) > MAX_REF_BYTES:
            raise HTTPException(413, "ref_audio exceeds size limit")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_file:
            ref_path = ref_file.name
            ref_file.write(raw)

    def _synthesize():
        from .model import ensure_loaded, generate  # lazy: needs torch/omnivoice
        ensure_loaded(MODEL_ID)
        return generate(text, ref_path, ref_text, instruct, guidance_scale, seed)

    try:
        loop = asyncio.get_event_loop()
        audio = await loop.run_in_executor(None, _synthesize)
    except (ImportError, ModuleNotFoundError) as exc:
        raise HTTPException(503, f"synthesis model not available in this build: {exc}") from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    finally:
        if ref_path is not None:
            os.unlink(ref_path)

    from .audio import pcm_to_ogg  # lazy: needs numpy/soundfile/ffmpeg
    return Response(content=pcm_to_ogg(audio), media_type="audio/ogg")


# Optionally serve the static frontend from this server (same origin → relative
# API calls just work). Mounted last so API routes win. Enable with SERVE_FRONTEND.
if _truthy(os.getenv("SERVE_FRONTEND")) and os.path.isdir(FRONTEND_DIR):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
