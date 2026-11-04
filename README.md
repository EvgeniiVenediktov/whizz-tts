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
| `GET /health` | status, bank path/word-count, model + Gemini flags |
| `GET /v1/languages` | `wordbank` + `phrasebank` (data-driven langs + counts) **and** `generate` (curated, verified list the model can generate) |
| `GET /v1/search?prefix=he` | **merged** word+phrase autocomplete; each result carries a `type` flag (`word`\|`phrase`) and its `voice` (`kind=word\|phrase` restricts) |
| `GET /v1/bank/{word}` · `/info` | serve a word (OGG default; `wav` dev-only) · sizes+type+voice |
| `GET /v1/phrase?text=…` · `/info` | serve a phrase (by text, optional `voice`) · sizes+type+voice+seed+reference |
| `POST /v1/tts` | live synthesis — `ref_audio` upload, a saved `voice=<name>`, or `instruct` (requires `MODEL_ENABLED`; else `503`) |
| `POST /v1/speak` | get-or-generate — `text` (required), `lang` (default `en-us`), `voice` (optional Gemini prebuilt, case-insensitive; unknown → `400`), `instruct` (optional). Bank hit in that voice → stored OGG, `X-Source: bank`. Miss → Gemini synth in that voice, saved into the phrase bank (`voice=gemini-<Voice>`), served with `X-Source: gemini`; `X-Voice` names the voice served. Needs `GEMINI_API_KEY` only on a miss |
| voice studio | `GET /v1/gemini/voices`, `POST /v1/gemini/tts`, `GET/POST /v1/voices`, `GET /v1/voices/{lang}/{name}/audio` — author + save clone references |

## Run

```bash
# Dev (server + frontend, same origin; bank only, no model):
SERVE_FRONTEND=1 .venv/bin/uvicorn src.server:app --port 8000   # → http://localhost:8000

# Docker (production path — plain docker build/run, no compose):
bash scripts/pack.sh              # → library.json + tts-bank.tar.gz
scripts/docker-build-fat.sh       # fat image: bank baked in, no torch
scripts/docker-run.sh fat

# scripts/docker-build-thin.sh builds code-only (no baked bank; needs a
# mounted /data volume, e.g. a k8s PVC); scripts/docker-run.sh thin runs it
# locally on a named /data volume, with library.json/tts-bank.tar.gz
# bind-mounted when present, else bank-less via GEMINI_API_KEY from .env:
scripts/docker-build-thin.sh && scripts/docker-run.sh thin   # DOCKER_DNS=8.8.8.8 if local DNS blocks Gemini

# Docker compose — dev-only convenience, mounts library.json/tts-bank.tar.gz as a volume:
docker compose --profile bank up --build   # lightweight bank-only (no torch)
docker compose --profile full up --build   # + live synthesis (downloads weights to a cache volume)
```

## Edge cache

An **edge** is the same image deployed next to a client (e.g. the website's
backend), holding its own copy of the bank. `/v1/speak` hits are served from
its volume; misses are fetched from the **upstream** server's `/v1/speak`,
saved, and served, so each new clip crosses the network once. The edge holds
no Gemini key, only a token for the upstream.

```
site backend --JWT(edge secret)--> edge (bank on /data) --miss, JWT(upstream secret)--> upstream (Gemini)
```

- Same request/voice semantics as the upstream; `X-Source: upstream` marks a
  clip just fetched. The voice is forwarded as requested, and the clip is filed
  under the voice upstream reports (`X-Voice`).
- `instruct` requests are forwarded and never saved.
- **Stateless upstream:** run the upstream with `SPEAK_STORE=0` and it never
  writes the bank — misses are generated with Gemini and returned only, so the
  clips live solely on the edges. It still serves hits from a bank if one is
  mounted. Concurrent identical misses share one Gemini call.
- Concurrent misses for one clip are fetched once (single-flight, also on the
  upstream for Gemini calls).
- Upstream `400` passes through; a rejected edge token, an upstream error, a
  non-OGG reply, or an unreachable upstream is `502`; a timeout is `504`.
  Nothing is saved on failure.

Set up (on the machine holding the upstream's `.env`):
```bash
python scripts/make_edge_env.py --upstream https://tts.example.com   # -> .env.edge (0600)
set -a; . ./.env.edge; set +a
python scripts/make_token.py --sub site --ttl 31536000   # the site backend's token for the edge
```
`.env.edge` gets `SPEAK_UPSTREAM_URL`, `SPEAK_UPSTREAM_TOKEN` (365 days by
default; re-run to renew; the edge logs a warning at startup within 30 days of
expiry) and the edge's **own** `JWT_SECRET`, kept across re-runs. Seed the bank
by putting `library.json` + `tts-bank.tar.gz` on the edge's `/data` (as for any
deployment), then run:
```bash
ENV_FILE=.env.edge scripts/docker-run.sh edge          # local; volume whizz-tts-edge-data
```
In production that is the thin image with `SPEAK_UPSTREAM_URL`,
`SPEAK_UPSTREAM_TOKEN`, `JWT_SECRET` and a persistent `/data`; one instance per
volume. The upstream must be reachable from the edge over HTTPS.

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
run the same check by hand:

```bash
python -m wordbank verify --desc library.json --root /data   # deployed (flat) layout
python -m wordbank verify --desc library.json                # dev checkout
```

`--root` selects the flat deploy layout (`/data/wordbank.db` + `/data/words_ogg`).
With no `--root`, `db`/`ogg_root`/`voices_root` fall back to the server's own env
defaults (`BANK_DB`/`OGG_ROOT`/`VOICES_ROOT`) — so a bare `verify` checks exactly
what the server would serve. Override any of them with `--db`/`--ogg-root`/`--voices-root`.

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
| `CORS_ORIGINS` | `*` (any) | comma-separated allowed origins. **Browser-enforced only — not authentication**; it does nothing against `curl` or any non-browser client |
| `JWT_SECRET` | unset | enables bearer-token auth on write/synthesis routes (see [Auth](#auth)); unset = open/dev |
| `JWT_AUDIENCE` / `JWT_ISSUER` | unset | optional — if set, tokens must carry a matching `aud`/`iss` claim |
| `SYNC_BUCKET` / `SYNC_PREFIX` | — | pull the bank from S3 at container start (structured + quoted; preferred over `BANK_SYNC_CMD`) |
| `BANK_SYNC_CMD` | — | escape hatch: arbitrary shell command run at container start to pull the bank into the volume. Runs via `sh -c` — **set it only from trusted configuration** |
| `SPEAK_UPSTREAM_URL` | unset | **edge mode**: `/v1/speak` misses are fetched from this whizz-tts server instead of Gemini (see [Edge cache](#edge-cache)) |
| `SPEAK_STORE` | `1` | `0` = `/v1/speak` never writes the bank (stateless upstream for edges); misses are generated and returned only. `/health` reports it as `store` |
| `SPEAK_UPSTREAM_TOKEN` | unset | bearer token for the upstream (signed with the upstream's `JWT_SECRET`; `scripts/make_edge_env.py` mints it) |
| `SPEAK_UPSTREAM_TIMEOUT` | `60` | seconds to wait for the upstream before answering `504` |

## Auth

The API is **open by default** (dev mode). Setting `JWT_SECRET` turns on
bearer-token auth for the write/synthesis routes — `POST /v1/tts`,
`POST /v1/gemini/tts`, `POST /v1/voices`, `POST /v1/speak` — via an HS256
`Authorization: Bearer <token>` check. Every `GET` endpoint and `/health` stay
public either way. A
missing/invalid/expired token gets a `401` with a `WWW-Authenticate: Bearer`
header and a JSON `detail`. Optional `JWT_AUDIENCE` / `JWT_ISSUER` add `aud`/
`iss` claim checks on top of the signature/expiry check.

Use a secret of at least 32 bytes — PyJWT warns on shorter HS256 keys (RFC
7518). `scripts/make_jwt_secret.py` generates one (64 CSPRNG bytes) straight
into `.env`, never printing it, and sets the file to `0600`; re-running keeps an
existing secret, `--rotate` replaces it (invalidating every issued token). Mint
a token with `scripts/make_token.py` (reads `JWT_SECRET` from the environment,
prints the token to stdout):

```bash
python scripts/make_jwt_secret.py
set -a; . ./.env; set +a
python scripts/make_token.py --sub dev --ttl 3600
```

```bash
curl -H "Authorization: Bearer $TOKEN" -F text="hello" http://localhost:8000/v1/tts -o out.ogg
```

`POST /v1/speak` is get-or-generate: a bank hit serves the stored OGG
(`X-Source: bank`); a miss synthesizes via Gemini, saves it into the phrase
bank (so the same text is a bank hit next time), and serves that
(`X-Source: gemini`) — it needs `GEMINI_API_KEY` only when it actually has to
generate. `voice` pins the recording: `voice=Puck` only ever serves or generates
Puck (a banked Kore word is not a hit). Without `voice`, any banked recording
serves (the default `GEMINI_TTS_VOICE` first) and a miss uses the default.
`instruct` (style direction) is a one-off render — never looked up or saved,
since the bank has no key for style.

```bash
curl -H "Authorization: Bearer $TOKEN" -F text="good morning" -F voice=Puck http://localhost:8000/v1/speak -o out.ogg
```

**Health and startup.** `/health` returns **503** when the service can serve
nothing — no bank, `MODEL_ENABLED` off, *and* no `GEMINI_API_KEY` — and 200
otherwise (a model-only or Gemini-only deployment with no bank is legitimate and
stays healthy; the first `/v1/speak` miss creates the bank on the volume). `docker-entrypoint.sh`
applies the same rule: it refuses to start (`exit 1`) rather than come up and
404 every lookup, and the image's `HEALTHCHECK` is wired to `/health` with a
start period generous enough to cover first-boot archive unpacking.

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
- [x] **Add JWT auth** — bearer tokens (HS256, `JWT_SECRET`) protect `/v1/tts`, `/v1/gemini/tts`, and `POST /v1/voices`; read-only bank endpoints stay public. See [Auth](#auth) below.
