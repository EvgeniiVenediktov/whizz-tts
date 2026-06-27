# TTS Word Bank

A multilingual text-to-speech service built around a **pre-generated word bank**:
audio for a large word list is generated once (cloud TTS), stored as compact
OGG, and served as static files with sub-millisecond lookup. Optional live
synthesis (OmniVoice) is available behind a flag for words not in the bank.

The design trades upfront generation for a tiny, cheap, low-latency runtime: the
serving image needs **no ML stack at all** unless live synthesis is enabled.

---

## Workflow

```
1. Local dev      generate audio (Gemini Batch) + code changes
                  → WAV masters + manifest, then encode OGG copies
2. Build + push   build the (code-only) image; push OGG + manifest to storage
3. Deploy         container pulls OGG + manifest into a volume and serves;
                  pulls model weights only when MODEL_ENABLED=1
```

## Architecture

| Part | What it is |
|------|------------|
| **Generation** (`wordbank/`) | Batch pipeline: word list → Gemini Batch API → WAV masters + SQLite manifest. Resumable, quota-aware. |
| **Encoding** (`scripts/encode_wavs.py`) | Mirrors WAV masters → OGG/Opus copy tree (16k VoIP, ~3 KB/word, ~20× smaller). WAV stays a local master; **OGG is what production serves**. |
| **Serving** (`src/server.py`) | FastAPI. Bank lookup (indexed SQLite) + static file serving. Optional live OmniVoice synth behind `MODEL_ENABLED`. |
| **Frontend** (`frontend/`) | Static UI: word-bank lookup (with autocomplete + WAV/OGG compare) and a synthesis form. Served by the API when `SERVE_FRONTEND=1`. |

## The manifest (`wordbank.db`)

A single SQLite table, `words`, is the source of truth (and the word→file index):

| Column | Notes |
|--------|-------|
| `id` | PK; also the filename prefix and shard key (`id // 1000`) |
| `word`, `lang` | `UNIQUE(word, lang)` → exact lookup is an O(log n) index hit |
| `status` | `pending → submitted → done` (or `verified` / `failed` / `quarantined`) |
| `file_path` | **relative to the bank root**, e.g. `en/gemini-Kore/1/1914_hello.wav` |
| `attempts`, `batch_id` | retry + async-batch bookkeeping |
| `transcript`, `billed_chars`, `error`, `created_at`, `updated_at` | metadata |

Audio lives in two parallel trees; the server resolves `file_path` under the
configured roots (so location is deployment config, not baked into the data):

```
$WAV_ROOT/en/gemini-Kore/1/1914_hello.wav   # masters (dev only)
$OGG_ROOT/en/gemini-Kore/1/1914_hello.ogg   # served copies
```

Lookup is plain SQLite — **no Elasticsearch or extra service.** Exact match uses
the `UNIQUE` index; prefix autocomplete uses an index range scan.

## API

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | status, bank path/word-count, model flag |
| `GET /v1/languages` | `wordbank` (data-driven: bank langs + counts) **and** `generate` (curated, verified list the model can generate, `wordbank/languages.py`) |
| `GET /v1/bank?prefix=he` | prefix autocomplete |
| `GET /v1/bank/{word}/info` | WAV/OGG sizes + availability (for the compare UI) |
| `GET /v1/bank/{word}?format=ogg` | serve the clip (OGG default; `wav` is dev-only) |
| `POST /v1/tts` | live synthesis (requires `MODEL_ENABLED`; else `503`) |

## Run

```bash
# Dev (server + frontend, same origin; bank only, no model):
SERVE_FRONTEND=1 .venv/bin/uvicorn src.server:app --port 8000   # → http://localhost:8000

# Docker — lightweight bank-only (no torch):
docker compose --profile bank up --build

# Docker — with live synthesis (downloads weights to a cache volume):
docker compose --profile full up --build
```

## Configuration (env)

| Var | Default | Purpose |
|-----|---------|---------|
| `BANK_DB` | `wordbank.db` | manifest path |
| `OGG_ROOT` | `output/words_ogg` | served OGG tree |
| `WAV_ROOT` | `output/words` | WAV masters (dev compare only) |
| `MODEL_ENABLED` | off | enable live `/v1/tts` synthesis |
| `PRELOAD_MODEL` | off | load weights at startup vs first request |
| `SERVE_FRONTEND` | off | serve the static UI from the API |
| `BANK_SYNC_CMD` | — | command run at container start to pull the bank into the volume |

## Generate / maintain the bank

```bash
python scripts/fetch_wordlist.py --lang en --n 50000          # credible word list (wordfreq)
GEMINI_API_KEY=… python -m wordbank batch-run --words data/words_en.txt --voice Kore
python scripts/encode_wavs.py output/words output/words_ogg   # OGG copies
python -m wordbank status                                     # progress by state
```
See [WORDBANK_PLAN.md](WORDBANK_PLAN.md) for the generation design and rationale.

## Tests

```bash
pytest                                              # unit (no model/cloud)
pytest --integration tests/test_integration.py      # loads OmniVoice (slow)
```

---

## TODO

- [ ] **Verify voice cloning** — exercise the OmniVoice `ref_audio` path end-to-end (it's wired but never fully validated against a real reference clip).
- [ ] **en-GB (and other accent) phrase banks via local OmniVoice** *(in progress)* — generate a one-off accent reference with Gemini (`scripts/make_accent_reference.py`), then have the local model clone it for the whole bank. See AGENTS.md → *Runbook: accent bank via local OmniVoice + Gemini reference*.
- [ ] **Add multi-language support** — same local-clone pattern per locale; serving + `/v1/languages` are already data-driven (language-scoped lookup + `(lang, word)` index), so it's mostly a generation + word-list task per language.
- [ ] **Add JWT auth** — the API is currently open; protect it behind bearer tokens (verify signature/expiry, optional scopes for `/v1/tts` vs read-only bank access).
