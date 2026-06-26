# TTS Server — Implementation Plan

## Goal
A lightweight, multilingual, high-quality TTS server based on
[k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) (0.6B params,
600+ languages, 40× real-time on GPU).

---

## Constraints & Decisions

| Concern | Decision |
|---------|----------|
| API | HTTP REST now; WebSocket streaming added later |
| Hardware | Auto-detect: CUDA → MPS → CPU |
| Packaging | Docker (multi-stage) |
| Voice cloning | Required — callers supply a reference `.wav` |
| Output format | OGG/Opus |
| Auth | None for now |

---

## Architecture

```
Client
  │  POST /v1/tts  (multipart or JSON + base64 ref audio)
  ▼
FastAPI app  (src/server.py)
  │
  ├── Request validation (Pydantic)
  ├── Device selector  (src/device.py)
  ├── Model singleton  (src/model.py)   ← OmniVoice loaded once at startup
  └── Audio encoder    (src/audio.py)   ← 24 kHz PCM → OGG/Opus via ffmpeg
```

---

## Phase 1 — Core server (MVP)

### 1.1 Project scaffold
```
tts/
├── src/
│   ├── server.py      # FastAPI app, lifespan, routes
│   ├── model.py       # OmniVoice singleton + generate()
│   ├── device.py      # auto-select cuda/mps/cpu
│   └── audio.py       # PCM → OGG/Opus conversion
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── PLAN.md
```

### 1.2 API surface (Phase 1)

**POST /v1/tts**
```
Content-Type: multipart/form-data

Fields:
  text          string   required   Text to synthesize
  ref_audio     file     required   Reference WAV for voice cloning
  ref_text      string   required   Transcript of ref_audio
  language      string   optional   BCP-47 tag; model auto-detects if omitted
  speed         float    optional   1.0 default

Response:
  Content-Type: audio/ogg
  Body: OGG/Opus audio at 24 kHz
```

**GET /health**
```
Response: {"status": "ok", "device": "cuda:0", "model": "k2-fsa/OmniVoice"}
```

### 1.3 Device selection logic (`src/device.py`)
```
if torch.cuda.is_available()  → "cuda:0"
elif torch.backends.mps.is_available() → "mps"
else → "cpu"
```

### 1.4 Model loading (`src/model.py`)
- Load once at FastAPI `lifespan` startup.
- `dtype=torch.float16` on CUDA, `torch.float32` on MPS/CPU.
- Expose a single `generate(text, ref_audio_path, ref_text) -> np.ndarray` function.

### 1.5 Audio encoding (`src/audio.py`)
- Write PCM to temp file, pipe through `ffmpeg -c:a libopus` to produce OGG.
- Return bytes to send as `StreamingResponse`.

### 1.6 Docker
- **Base image**: `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` (CUDA) or
  `python:3.11-slim` with CPU-only torch for a smaller CPU-only variant.
- Two compose profiles: `gpu` and `cpu`.
- `ffmpeg` installed via `apt`.
- Model weights downloaded at first run (cached in a Docker volume).

---

## Phase 2 — Quality & reliability

| Item | Detail |
|------|--------|
| Request queue | `asyncio.Semaphore` to limit concurrent inferences (default: 1 on CPU, configurable) |
| Temp file cleanup | Context manager; always deleted after response |
| Input validation | Max text length (configurable), ref audio size limit |
| Structured logging | JSON logs with request_id, latency, device |
| Env-based config | `MAX_CONCURRENCY`, `MAX_TEXT_LEN`, `HF_HOME`, `LOG_LEVEL` via `.env` |

---

## Phase 3 — WebSocket streaming (future)

**WS /v1/tts/stream**
- Client sends JSON frame with text + ref audio (base64).
- Server yields OGG/Opus chunks as they come out of the model.
- Requires confirming whether OmniVoice supports incremental/streaming decode;
  fallback: stream the final file in chunks.

---

## Open questions (resolve before Phase 2)

1. Does OmniVoice support CPU inference at usable speed? (RTF 0.025 is GPU-measured.)
2. MPS (Apple Silicon) — needs testing; float16 may need to be forced to float32.
3. Model weights size on disk (~1.2 GB estimated for 0.6B fp16) — confirm for volume sizing.

---

## Implementation order

1. `src/device.py` + `src/model.py` — load model, run a smoke test locally
2. `src/audio.py` — PCM → OGG/Opus
3. `src/server.py` — FastAPI routes
4. `Dockerfile` + `docker-compose.yml`
5. Manual end-to-end test (curl)
6. Phase 2 hardening
