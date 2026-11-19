# TTS Word Bank

A multilingual text-to-speech service built around a **pre-generated word bank**:
audio for a large word list is generated once (cloud TTS), stored as compact
OGG, and served as static files with sub-millisecond lookup. Anything not in the
bank is generated on demand with Gemini (`/v1/speak`) and saved into it.

The design trades upfront generation for a tiny, cheap, low-latency runtime: the
serving image needs **no ML stack at all**.

---

## Workflow

```
1. Local dev      generate audio (Gemini Batch) + code changes
                  → WAV masters + manifest, then encode OGG copies
2. Build + push   build the (code-only) image; push OGG + manifest to storage
3. Deploy         container pulls OGG + manifest into a volume and serves
```

## Architecture

| Part | What it is |
|------|------------|
| **Generation** (`wordbank/`) | Batch pipeline: word list → Gemini Batch API → WAV masters + SQLite manifest. Resumable, quota-aware. |
| **Encoding** (`scripts/encode_wavs.py`) | Mirrors WAV masters → OGG/Opus copy tree (16k VoIP, ~3 KB/word, ~20× smaller). WAV stays a local master; **OGG is what production serves**. |
| **Serving** (`src/server.py`) | FastAPI. Bank lookup (indexed SQLite) + static file serving; `/v1/speak` fills misses via Gemini. |
| **Frontend** (`frontend/`) | Static UI: bank lookup (autocomplete, WAV/OGG compare, Speak) and a Gemini voice preview. Served by the API when `SERVE_FRONTEND=1`. |

## The manifest (`wordbank.db`)

Two content tables are the source of truth (and the text→file index), stored
**separately so words and phrases never overlap**:

- **`words`** — the Gemini single-word bank, `UNIQUE(lang, word, voice)`: one
  row per clip, so a word can be banked in several voices (e.g. a woman's and a
  man's voice per language). Banks keyed `UNIQUE(word, lang)` are rebuilt once
  when the `wordbank` tools (or the server's first `/v1/speak` save) open them:
  rows and ids kept, a missing `voice` recovered from `file_path`, and words
  not generated yet given their language's voice when it has exactly one. A
  NULL `voice` left after that marks a word never generated: the first voice
  to generate it claims the row.
- **`phrases`** — the phrase bank, `UNIQUE(text, lang, voice)` (a phrase can
  exist in several voices). Written by `POST /v1/speak` on a miss. Its `voice`
  is the same label as a word's (`Kore`), so a voice's bank is its words plus
  its phrases. Phrases saved before that say `gemini-Kore`: relabelled once
  when the `wordbank` tools (or the server's first `/v1/speak` save) open the
  bank, and read as `Kore` by lookups until then. A clash with a row already
  labelled `Kore` for the same text keeps the one with a clip, then the older;
  the other is deleted if it has no clip, else kept under the old spelling.

Both share the same columns:

| Column | Notes |
|--------|-------|
| `id` | PK; also the filename prefix and shard key (`id // 1000`) |
| `word` / `text`, `lang` | leading `(lang, …)` index → exact lookup is an O(log n) hit |
| `voice` | the voice label the clip was generated with (`Kore`, `en-gb-tutor-11`; the same for words and phrases; surfaced in lookups) |
| `status` | `pending → submitted → done` (or `verified` / `failed` / `quarantined`; `retake`: a batch take QC flagged, queued for a re-take, not served) |
| `file_path` | **relative to the bank root**, e.g. `en/gemini-Kore/1/1914_hello.wav` |
| `seed`, `ref_audio`, `ref_text` | provenance of phrases made by the removed local OmniVoice pipeline; null for Gemini clips |
| `model`, `style` | **provenance for Gemini clips** — the model and the delivery direction used (`style`: the speech-metadata style on 3.8 models, the inline prefix such as `VOICE_INSTRUCTION` on 3.1/2.5; NULL = none). NULL on rows generated before these columns existed. Shown by `wordbank export` |
| `review` | why QC (`wordbank/qc.py`) flagged the clip, e.g. `ends on sound (-31 dB)`, `too long (3.40 s)`; NULL = passed or never checked |
| `qc_seconds`, `qc_end_db`, `qc_takes` | QC measurements of the current clip (length, last-100 ms level in dBFS; NULL = never checked) and the re-takes it took |
| `attempts`, `batch_id`, `transcript`, `billed_chars`, `error`, `created_at`, `updated_at` | bookkeeping + metadata (`error`: the last generation failure) |

Banks created before the local pipeline was removed may also hold an unused
`voices` table.

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
| `GET /health` | status, bank path/word-count, Gemini/upstream/store flags |
| `GET /v1/languages` | `wordbank` + `phrasebank` (data-driven langs + counts) **and** `generate` (curated, listen-verified list) |
| `GET /v1/search?prefix=he` | **merged** word+phrase autocomplete; each result carries a `type` flag (`word`\|`phrase`) and its `voice`, one per voice banked (`kind=word\|phrase`, `voice` restrict) |
| `GET /v1/bank?prefix=he` | word-only autocomplete (index range scan; optional `voice`) |
| `GET /v1/bank/{word}` · `/info` | serve a word (OGG default; `wav` dev-only; optional `voice`) · sizes+type+voice |
| `GET /v1/phrase?text=…` · `/info` | serve a phrase (by text, optional `voice`) · sizes+type+voice+provenance |
| `POST /v1/speak` | get-or-generate — `text` (required), `lang` (default `en-us`), `voice` (optional: a classic Gemini voice or a curated locale voice, case-insensitive; unknown → `400`), `instruct` (optional, one-off: not looked up or saved). Bank hit in that voice → stored OGG, `X-Source: bank`. Miss → Gemini synth in that voice, saved into the phrase bank under the voice's label, as words are (`Kore`, `en-gb-tutor-11`; filed in `<lang>/gemini-<label>/phrases/`), served with `X-Source: gemini`; `X-Voice` names the voice served (that label, for a word or a phrase). Needs `GEMINI_API_KEY` only on a miss |
| voice studio | `GET /v1/gemini/voices` (the 30 classic voices), `GET /v1/gemini/languages` (the curated female + male voice per language), `POST /v1/gemini/tts` (preview a voice, nothing saved) |

### Endpoint reference

Read-only endpoints are public. The `POST` routes marked **JWT** require
`Authorization: Bearer <token>` when `JWT_SECRET` is set (see [Auth](#auth));
with it unset the whole API is open (dev mode).

`lang` (every endpoint that takes it) is a language code, default `en-us`,
case-insensitive and stored lowercase (`EN-US` = `en-us`). Anything else
(`../x`, an absolute path, `en_us`, non-ASCII, an empty query value) is a
**400** `invalid lang '…'`; on `POST` routes an empty form field means the
default. In edge mode, `/v1/speak` answers **502** if the upstream's `X-Voice`
isn't a single path segment.

#### Read-only (bank + status)

**`GET /health`** — no params. `{status, healthy, gemini, upstream, store, bank,
bank_path, bank_words}`. `200` if the service can serve *something* (a bank, a
Gemini key, or an upstream), else `503`.

**`GET /v1/languages`** — no params. `{wordbank, phrasebank, generate}`:
the first two are data-driven `{code, words}` counts from the bank; `generate`
is the curated, listen-verified list generation supports.

**`GET /v1/search`** — `prefix` (required) · `lang` (default `en-us`) ·
`limit` (default 10, capped 50) · `kind` (`word`|`phrase`, optional filter) ·
`voice` (optional: words and phrases in that voice).
Merged prefix search; each result is `{text, type: word|phrase, voice}`, one
per voice a text is banked in.

**`GET /v1/bank`** — `prefix` · `lang` · `limit` (same defaults) · `voice`
(optional). Word-only autocomplete → `{words: [...]}` (distinct words).

**`GET /v1/bank/{word}`** — `lang` (default `en-us`) · `format` (`ogg` default;
`wav` serves the dev master and 404s in production) · `voice` (optional,
case-insensitive label: `Kore`, `en-gb-tutor-14`). The word's audio in that
voice; without `voice`, the default `GEMINI_TTS_VOICE`'s clip if banked, else
the earliest banked one. 404 `not in bank` otherwise.
**`GET /v1/bank/{word}/info`** — same params; `{word, type, voice, wav: {exists,
bytes}, ogg: {...}}` for the compare UI.

**`GET /v1/phrase`** — `text` (required) · `lang` · `voice` (optional — pin a
specific voice by its label, `Kore`; the old `gemini-Kore` is accepted too;
omitted = any) · `format`. The phrase's audio, or 404.
**`GET /v1/phrase/info`** — same params, plus provenance in the response:
`seed`, `ref_audio`, `ref_text` (set only on phrases from the removed local
pipeline).

**`GET /v1/gemini/voices`** — no params. The 30 classic Gemini voices
`{name, style, gender}`. They are style presets, **not** language-bound —
Gemini auto-detects language from the text — and work on every TTS model.

**`GET /v1/gemini/languages`** — no params. The curated voices per language
(`wordbank/gemini_voices.json`): one female and one male voice for each of
`en-us`, `en-gb`, `ru-ru`, `fr-fr`, `de-de`, `zh-cn`, `ja-jp`, `es-es` (bank
`lang` codes), with their Gemini voice-library metadata, plus
`locale_voice_models` (the models that take locale voices):
```json
{"languages": [
  {"code": "en-gb", "name": "English (UK)", "voices": [
    {"name": "en-gb-tutor-11", "classic": false, "gender": "female",
     "display_name": "Tutor 11", "language_code": "en-GB", "accent": "Winchester English",
     "pitch": "medium", "persona": "Educational Tutor (Music Teacher)", "description": "…"},
    {"name": "en-gb-tutor-14", "classic": false, "gender": "male", "…": "…"}]},
  {"code": "en-us", "name": "English (US)", "voices": [
    {"name": "Kore", "classic": true, "…": "…"}, {"name": "Iapetus", "classic": true, "…": "…"}]},
  "…"],
 "locale_voice_models": ["gemini-3.8-flash-lite-tts", "gemini-3.8-flash-tts"]}
```
`name` is what `voice` takes. `en-us` uses two classic voices (`classic: true`,
language-agnostic); the others are **locale voices**, which carry their accent
and are verified only on the 3.8 models: on 3.1/2.5 they are untested, so a
request pairing one with those models is refused (`400`) before any Gemini call.
The curated voices are not the `generate` list of `/v1/languages` (no bank is
implied). Refresh their metadata, or list candidates for a new pick, with
`scripts/fetch_gemini_voices.py`.

#### Write / synthesis (JWT, multipart form fields)

**`POST /v1/speak`** — get-or-generate. `text` (required, ≤ `MAX_TEXT_LEN`) ·
`lang` (default `en-us`) · `voice` (a classic voice from `/v1/gemini/voices`
or a curated locale voice from `/v1/gemini/languages`, case-insensitive; default
`GEMINI_TTS_VOICE` = Kore; also pins the bank lookup: words and phrases by the
same label, where a classic label keeps its capitals (`Kore`)
and a locale voice is its lowercase id (`en-gb-tutor-11`); any language may use
any of them) · `instruct` (delivery direction). Bank hit (words table for single tokens, phrases otherwise) → stored
OGG with `X-Source: bank`; miss → Gemini synth, persisted into the phrase bank,
returned with `X-Source: gemini`. A request with `instruct` is a **one-off**:
never looked up or saved (the bank has no key for it). How `instruct` reaches
Gemini depends on `GEMINI_TTS_MODEL`: the 3.8 models get it as the
speech-metadata **style**, with `text` sent verbatim (they read an inline prefix
aloud); 3.1/2.5 get it as an inline prefix (`"<instruct>: <text>"`). Misses
without `instruct` are generated with `GEMINI_TTS_STYLE` when set and saved with
it (`model`/`style` columns); the one-off's `instruct` replaces it rather than
adding to it. Errors: `400` empty/oversize text, unknown `voice`, or a locale
`voice` when `GEMINI_TTS_MODEL` isn't a 3.8 model (misses and one-offs only: a
banked clip in that voice still serves), `500` `GEMINI_TTS_STYLE` set with
a model that takes no style (misses only; hits and `instruct` one-offs still
work) or a default `GEMINI_TTS_VOICE` that is a locale voice on a non-3.8
model, `502` Gemini failure (detail names the `finish_reason`, or the
Interactions `status`, when it returned no audio), `503` encoding stack absent.

**`POST /v1/gemini/tts`** — studio preview. `text` (required, ≤ `MAX_TEXT_LEN`)
· `voice` (as in `/v1/speak`, default `GEMINI_TTS_VOICE`; a locale voice with a
non-3.8 `model` → `400`) · `instruct` (direction, routed per model
as in `/v1/speak`) · `style` (speech-metadata style; 3.8 models only, else `400`;
with `instruct` too, the style comes first) · `model` (default
`GEMINI_TTS_MODEL`). Synthesizes one phrase with Gemini and returns **WAV**
(never banked). `GEMINI_TTS_STYLE` does not apply. Needs `GEMINI_API_KEY`;
`502` on upstream failure.

## Run

```bash
# Dev (server + frontend, same origin):
SERVE_FRONTEND=1 .venv/bin/uvicorn src.server:app --port 8000   # → http://localhost:8000

# Docker (production path — plain docker build/run, no compose):
bash scripts/pack.sh              # → library.json + tts-bank.tar.gz
scripts/docker-build-fat.sh       # fat image: bank baked in
scripts/docker-run.sh fat

# scripts/docker-build-thin.sh builds code-only (no baked bank; needs a
# mounted /data volume, e.g. a k8s PVC); scripts/docker-run.sh thin runs it
# locally on a named /data volume, with library.json/tts-bank.tar.gz
# bind-mounted when present, else bank-less via GEMINI_API_KEY from .env:
scripts/docker-build-thin.sh && scripts/docker-run.sh thin   # DOCKER_DNS=8.8.8.8 if local DNS blocks Gemini

# Docker compose — dev-only convenience, mounts library.json/tts-bank.tar.gz as a volume:
docker compose --profile bank up --build
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
  under the voice upstream reports (`X-Voice`), as its label: an upstream on
  older code reports a phrase as `gemini-Kore`, filed here as `Kore`.
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
# → library.json     description: languages, word (clip) / phrase counts, per-voice word counts, folders
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
With no `--root`, `db`/`ogg_root` fall back to the server's own env defaults
(`BANK_DB`/`OGG_ROOT`) — so a bare `verify` checks exactly what the server would
serve. Override either with `--db`/`--ogg-root`.

**Bare-metal (no Docker)** — unpack the archive and run directly:

```bash
mkdir tts-bank && tar xzf tts-bank.tar.gz -C tts-bank && cd tts-bank
bash scripts/unpack.sh                     # .venv + serving deps, verifies the bank
SERVE_FRONTEND=1 .venv/bin/uvicorn src.server:app --port 8000
```

## Configuration (env)

| Var | Default | Purpose |
|-----|---------|---------|
| `BANK_DB` | `wordbank.db` | manifest path |
| `OGG_ROOT` | `output/words_ogg` | served OGG tree |
| `WAV_ROOT` | `output/words` | WAV masters (dev compare only) |
| `SERVE_FRONTEND` | off | serve the static UI from the API |
| `CORS_ORIGINS` | `*` (any) | comma-separated allowed origins. **Browser-enforced only — not authentication**; it does nothing against `curl` or any non-browser client |
| `JWT_SECRET` | unset | enables bearer-token auth on write/synthesis routes (see [Auth](#auth)); unset = open/dev |
| `JWT_AUDIENCE` / `JWT_ISSUER` | unset | optional — if set, tokens must carry a matching `aud`/`iss` claim |
| `SYNC_BUCKET` / `SYNC_PREFIX` | — | pull the bank from S3 at container start (structured + quoted; preferred over `BANK_SYNC_CMD`) |
| `BANK_SYNC_CMD` | — | escape hatch: arbitrary shell command run at container start to pull the bank into the volume. Runs via `sh -c` — **set it only from trusted configuration** |
| `GEMINI_TTS_MODEL` | `gemini-3.8-flash-tts` | Gemini TTS model for `/v1/speak` misses and the studio default |
| `GEMINI_TTS_VOICE` | `Kore` | default voice for `/v1/speak` and the studio |
| `GEMINI_TTS_STYLE` | unset | deployment-wide delivery direction for every `/v1/speak` miss (not `instruct` one-offs, not the studio), sent as the speech-metadata style with the text verbatim, and saved as the clip's `style`. Needs a style-capable `GEMINI_TTS_MODEL` (`gemini-3.8-flash-tts`, `gemini-3.8-flash-lite-tts`): with another model the server warns at startup and misses answer `500`. Ignored in edge mode. Changing it doesn't touch banked clips |
| `GEMINI_TTS_EMPTY_RETRIES` | `2` | extra attempts when Gemini TTS returns no audio with a transient `finish_reason` (`OTHER`/unset); blocks (`SAFETY`, `PROHIBITED_CONTENT`, a prompt `block_reason`, ...) are never retried. A final empty response is a `502` naming the reason |
| `SPEAK_UPSTREAM_URL` | unset | **edge mode**: `/v1/speak` misses are fetched from this whizz-tts server instead of Gemini (see [Edge cache](#edge-cache)) |
| `SPEAK_STORE` | `1` | `0` = `/v1/speak` never writes the bank (stateless upstream for edges); misses are generated and returned only. `/health` reports it as `store` |
| `SPEAK_UPSTREAM_TOKEN` | unset | bearer token for the upstream (signed with the upstream's `JWT_SECRET`; `scripts/make_edge_env.py` mints it) |
| `SPEAK_UPSTREAM_TIMEOUT` | `60` | seconds to wait for the upstream before answering `504` |

## Auth

The API is **open by default** (dev mode). Setting `JWT_SECRET` turns on
bearer-token auth for the synthesis routes — `POST /v1/speak` and
`POST /v1/gemini/tts` — via an HS256
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

`POST /v1/speak` is get-or-generate: a bank hit serves the stored OGG
(`X-Source: bank`); a miss synthesizes via Gemini, saves it into the phrase
bank (so the same text is a bank hit next time), and serves that
(`X-Source: gemini`) — it needs `GEMINI_API_KEY` only when it actually has to
generate. `voice` pins the recording: `voice=Puck` only ever serves or generates
Puck (a banked Kore word is not a hit). Without `voice`, any banked recording
serves (the default `GEMINI_TTS_VOICE` first) and a miss uses the default.
`instruct` (style direction) is a one-off render — never looked up or saved,
since the bank has no key for style; on the 3.8 models it is sent as the
speech-metadata style, so it is never spoken.

```bash
curl -H "Authorization: Bearer $TOKEN" -F text="good morning" -F voice=Puck http://localhost:8000/v1/speak -o out.ogg
```

**Health and startup.** `/health` returns **503** when the service can serve
nothing — no bank, no `GEMINI_API_KEY`, *and* no `SPEAK_UPSTREAM_URL` — and 200
otherwise (a Gemini-only or edge deployment with no bank is legitimate and
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

# Word bank with a delivery style (3.8 models only; keeps them from acting out
# "haha", "laughing", "russian"). Batch carries it as speechMetadata (documented,
# smoke-test first: scripts/gemini_batch_smoke.py --model … --style "…"):
GEMINI_API_KEY=… python -m wordbank --lang en-gb batch-run --words words.txt \
    --model gemini-3.8-flash-lite-tts --voice en-gb-tutor-11 --style "plain and neutral"
# …or synchronously (Interactions API, verified), rate-limited, resumable:
GEMINI_API_KEY=… python -m wordbank --lang en-gb gemini-run --words words.txt \
    --model gemini-3.8-flash-lite-tts --voice en-gb-tutor-11 --style "plain and neutral" --rpm 10

# A second voice for the same words (a man's bank next to a woman's): new rows,
# filed under gemini-en-gb-tutor-14; the en-gb-tutor-11 clips are untouched:
GEMINI_API_KEY=… python -m wordbank --lang en-gb batch-run --words words.txt \
    --model gemini-3.8-flash-lite-tts --voice en-gb-tutor-14 --style "plain and neutral"

# Redo some words: move their current take to <db dir>/takes/<UTC>/{wav,ogg}/
# (+ words.json), reset them to pending, generate again, re-encode
# (--voice: only that voice's rows; default: every voice the word is banked in):
python -m wordbank --lang en-gb regenerate --words-file redo.txt --ogg-root output/words_ogg
GEMINI_API_KEY=… python -m wordbank --lang en-gb gemini-run --model gemini-3.8-flash-tts \
    --voice Kore --style "plain and neutral"
python scripts/encode_wavs.py output/words output/words_ogg

# QC an existing bank (reads the WAV masters): record flags, list them for a listen,
# then re-take the flagged words until they pass (2 extra takes each, the best kept):
python -m wordbank --lang en-gb qc --voice en-gb-tutor-11 --csv review.csv
GEMINI_API_KEY=… python -m wordbank --lang en-gb qc --voice en-gb-tutor-11 --retry \
    --model gemini-3.8-flash-lite-tts --style "plain and neutral" --ogg-root output/words_ogg \
    --csv review.csv
python scripts/encode_wavs.py output/words output/words_ogg
python -m wordbank status                                     # flag / never-checked counts
```
The phrase bank fills itself: every `/v1/speak` miss is generated and saved.

Directions per model (`gemini_tts.TTS_MODELS`): the 3.8 models take a `--style`
and read any inline instruction aloud, so `batch*`/`gemini-run` never send them
one — an explicit `--instruction` becomes the style, and with neither flag they
refuse rather than default to `VOICE_INSTRUCTION` (`--instruction=` = the bare
word). 3.1/2.5 keep the inline `--instruction` (default `VOICE_INSTRUCTION`) and
reject `--style`. `--voice` takes a classic name (`Kore`) or any voice-library
id (`en-gb-tutor-11`, filed as `gemini-en-gb-tutor-11`); a non-classic voice
with a non-3.8 `--model` (the batch default is 3.1) exits before anything is
ingested or sent. `gemini-run` takes `--rpm` (default 10, Tier 1),
`--max-attempts` (3) and `--limit`; it retries 429/5xx with backoff and stops if
they persist, or after 5 rows in a row fail otherwise.

Every command ingests `--words` for its `--voice` (case-insensitive, stored as
the bank label: `Kore`, `en-gb-tutor-11`), so the same word list in a second
voice makes new rows instead of being skipped, and only that voice's rows are
generated.

**Clip QC** (`wordbank/qc.py`, stdlib only). A clip is flagged when its last
100 ms is louder than -40 dBFS (it ends on sound — usually cut off) or its
length is outside 0.25–3.0 s. `gemini-run`, `batch`, `batch-collect` and
`batch-run` check every clip and re-take a flagged one up to `--qc-retries`
times (default 2; 0 = only flag), with the same model, voice and style: the
first passing take is kept; if every take is flagged the best one is kept
(plausible length first, then the quietest end) and its reason stays in
`review` for a listen. `gemini-run` re-takes inline (within `--rpm`); a batch
re-queues the row (status `retake`, its best take held in `<db dir>/takes/qc/`,
outside the bank tree) for the next submission with the same `--voice`,
`--model` and `--style` — the next wave of `batch-run`, or the next
`batch-submit`/`batch-run`. Re-takes don't count toward `--max-attempts`.
`wordbank qc` checks an existing bank (`--voice`, `--all-langs`, `--csv`) and
with `--retry` re-takes flagged words synchronously (`--model`, `--style`,
`--rpm`, `--qc-retries`; `--ogg-root` moves a replaced clip's OGG aside so it
gets re-encoded). `--accept-file keep.txt` records words a person listened to and
kept despite a flag (`qc_accepted`): their flag is cleared and later `qc` runs skip
them, until a `regenerate` gives the word a new take. Phrases made by `/v1/speak`
are not checked.

## Tests

```bash
pytest                                              # unit (no cloud)
set -a; . ./.env; set +a
pytest --integration tests/test_integration.py      # live Gemini (paid; one synthesis)
```

---

## TODO

- [ ] **Add multi-language support** — serving + `/v1/languages` are already data-driven (language-scoped lookup + `(lang, word)` index), so it's a word list + a Gemini batch run + a verified listen per language.
- [x] **Add JWT auth** — bearer tokens (HS256, `JWT_SECRET`) protect `/v1/speak` and `/v1/gemini/tts`; read-only bank endpoints stay public. See [Auth](#auth) above.
