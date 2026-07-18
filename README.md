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

Two content tables are the source of truth (and the text→file index), stored
**separately so words and phrases never overlap**:

- **`words`** — the Gemini single-word bank, `UNIQUE(word, lang)`.
- **`phrases`** — the local OmniVoice phrase bank, `UNIQUE(text, lang, voice)`
  (a phrase can exist in several voices). Written by `wordbank generate`.

Both share the same columns:

| Column | Notes |
|--------|-------|
| `id` | PK; also the filename prefix and shard key (`id // 1000`) |
| `word` / `text`, `lang` | leading `(lang, …)` index → exact lookup is an O(log n) hit |
| `voice` | the voice/reference label the clip was generated with (surfaced in lookups) |
| `status` | `pending → submitted → done` (or `verified` / `failed` / `quarantined`) |
| `file_path` | **relative to the bank root**, e.g. `en/gemini-Kore/1/1914_hello.wav` |
| `seed`, `ref_audio`, `ref_text` | **provenance for local phrases** — the RNG seed and the clone reference used, so every generated clip is reproducible |
| `attempts`, `batch_id`, `transcript`, `billed_chars`, `error`, `created_at`, `updated_at` | bookkeeping + metadata |

A third table, **`voices`**, is a registry of reusable clone references
(`name, lang, ref_audio, ref_text, …`) — generation-side only, never served.

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
| `GET /v1/languages` | `wordbank` + `phrasebank` (data-driven langs + counts) **and** `generate` (curated, verified list the model can generate) |
| `GET /v1/search?prefix=he` | **merged** word+phrase autocomplete; each result carries a `type` flag (`word`\|`phrase`) and its `voice` (`kind=word\|phrase` restricts) |
| `GET /v1/bank/{word}` · `/info` | serve a word (OGG default; `wav` dev-only) · sizes+type+voice |
| `GET /v1/phrase?text=…` · `/info` | serve a phrase (by text, optional `voice`) · sizes+type+voice+seed+reference |
| `POST /v1/tts` | live synthesis — `ref_audio` upload, a saved `voice=<name>`, or `instruct` (requires `MODEL_ENABLED`; else `503`) |
| voice studio | `GET /v1/gemini/voices`, `POST /v1/gemini/tts`, `GET/POST /v1/voices`, `GET /v1/voices/{lang}/{name}/audio` — author + save clone references |

## Run

```bash
# Dev (server + frontend, same origin; bank only, no model):
SERVE_FRONTEND=1 .venv/bin/uvicorn src.server:app --port 8000   # → http://localhost:8000

# Docker — lightweight bank-only (no torch):
docker compose --profile bank up --build

# Docker — with live synthesis (downloads weights to a cache volume):
docker compose --profile full up --build
```

## Deployment package (image + description + archive)

The image is **code only** — it never stores the bank. `scripts/pack.sh` produces
the two data artifacts that ride alongside it:

```bash
bash scripts/pack.sh
# → library.json     description: languages, voices, word/phrase counts, folders
# → tts-bank.tar.gz  library code + wordbank.db + served OGG (excludes WAV masters)
```

**Deploy = the Docker image + `library.json` + `tts-bank.tar.gz`.** Put the
description and archive on the (persistent) volume; on first boot the container
verifies the volume against `library.json` — described folders present, word/phrase
counts and OGG count match — and unpacks the archive only if it isn't provisioned.
Later restarts find the volume ready and skip the unpack.

```bash
docker compose --profile bank up --build   # mounts library.json + tts-bank.tar.gz on /data
```

Provisioning is `docker-entrypoint.sh` calling `python -m wordbank verify`; you can
run the same check by hand: `python -m wordbank verify --desc library.json --root /data`.

**Bare-metal (no Docker)** — unpack the archive and run directly:

```bash
mkdir tts-bank && tar xzf tts-bank.tar.gz -C tts-bank && cd tts-bank
bash scripts/unpack.sh                     # .venv + serving deps (no ML stack), verifies the bank
SERVE_FRONTEND=1 .venv/bin/uvicorn src.server:app --port 8000
```

## Configuration (env)

| Var | Default | Purpose |
|-----|---------|---------|
| `BANK_DB` | `wordbank.db` | manifest path |
| `OGG_ROOT` | `output/words_ogg` | served OGG tree |
| `WAV_ROOT` | `output/words` | WAV masters (dev compare only) |
| `VOICES_ROOT` | `output/voices` | clone-reference clips (voice studio) |
| `MODEL_ENABLED` | off | enable live `/v1/tts` synthesis |
| `PRELOAD_MODEL` | off | load weights at startup vs first request |
| `SERVE_FRONTEND` | off | serve the static UI from the API |
| `BANK_SYNC_CMD` | — | command run at container start to pull the bank into the volume |

## Generate / maintain the bank

```bash
# Word bank (Gemini Batch):
python scripts/fetch_wordlist.py --lang en --n 50000          # credible word list (wordfreq)
GEMINI_API_KEY=… python -m wordbank batch-run --words data/words_en.txt --voice Kore
python scripts/encode_wavs.py output/words output/words_ogg   # OGG copies
python -m wordbank status                                     # progress (words + phrases)

# Phrase bank (local OmniVoice; accent via a Gemini clone reference):
GEMINI_API_KEY=… python -m wordbank --lang en-gb gemini-voice --text "…" --name en-gb  # make+save a reference
python -m wordbank --lang en-gb generate --words phrases.txt --voice en-gb  # seed + reference recorded per phrase
```
Every locally-generated phrase records the RNG **seed** and the **reference clip**
used, so it's reproducible. See [WORDBANK_PLAN.md](WORDBANK_PLAN.md) for rationale.

## Tests

```bash
pytest                                              # unit (no model/cloud)
pytest --integration tests/test_integration.py      # loads OmniVoice (slow)
```

---

## TODO

- [ ] **Validate voice-cloning quality** — the local phrase pipeline is wired (records seed + reference per clip) but the cloning *quality* hasn't been judged against a real accent reference. Generate an en-GB set and listen.
- [ ] **Fill the phrase bank** — the `phrases` table, merged `/v1/search`, and the voice studio all work; what's left is authoring reference voices and generating phrase sets per accent. See AGENTS.md → *Runbook: accent bank via local OmniVoice + Gemini reference*.
- [ ] **Add multi-language support** — same local-clone pattern per locale; serving + `/v1/languages` are already data-driven (language-scoped lookup + `(lang, word)` index), so it's mostly a generation + word-list task per language.
- [ ] **Add JWT auth** — the API is currently open; protect it behind bearer tokens (verify signature/expiry, optional scopes for `/v1/tts` vs read-only bank access).
