# Agent guide

Orientation, dev setup, invariants, and runbooks. Full detail in `README.md`.

## Project at a glance
- **Generate** → WAV masters in `output/words/` + manifest `wordbank.db` via the **Gemini Batch** path (`gemini_batch.py` → the `words` table). Phrases go into the **separate `phrases` table**, filled by `/v1/speak` misses. Words and phrases are stored in distinct tables so they never overlap. (Local OmniVoice generation, voice cloning and the clone-reference registry were removed; `phrases.seed`/`ref_audio`/`ref_text` remain only as legacy provenance columns.)
- **Encode** (`scripts/encode_wavs.py`) → OGG copies in `output/words_ogg/` (16k VoIP, ~3 KB/word).
- **Serve** (`src/server.py`, FastAPI) → indexed bank lookup over the manifest + static OGG. `GET /v1/search` is the merged word+phrase lookup — each result carries a `type` flag (`word`|`phrase`) and the `voice` it was made with; phrases serve from `/v1/phrase`. `POST /v1/speak` is **get-or-generate**: a bank hit (word or phrase table) serves the stored OGG (`X-Source: bank`); a miss synthesizes via Gemini, saves it into the phrase bank, and serves that (`X-Source: gemini`) — so the same text is a bank hit next time. `voice` (Gemini prebuilt, case-insensitive, unknown → 400) **pins** the recording: only that voice is served or generated (words match the bare name, falling back to the voice in `file_path` for un-backfilled rows; phrases match `gemini-<Voice>`). No `voice` → any banked recording, the default voice's first; a miss uses the default. `instruct` makes it a one-off (no lookup, no save — the bank has no style key), routed per `GEMINI_TTS_MODEL` (3.8: the speech_metadata style; 3.1/2.5: inline prefix). `GEMINI_TTS_STYLE` (unset by default) is a deployment-wide style for every other miss, saved with the clip as provenance. `X-Voice` reports the voice label served. Saved clips use the bank OGG profile (`BANK_OGG_*` in `wordbank/constants.py`) under `<lang>/gemini-<Voice>/phrases/`. Also hosts the **voice studio**: `GET /v1/gemini/voices` (the 30 prebuilt Gemini voices `/v1/speak` accepts — downstream clients read the list from here) and `POST /v1/gemini/tts` (preview a phrase in a voice, nothing saved). Gemini voices are **style presets, not language-bound** — language is auto-detected from the text.
- **Edge mode** (`SPEAK_UPSTREAM_URL` + `SPEAK_UPSTREAM_TOKEN`, `src/upstream.py`): the same image deployed next to a client as a cache: its own bank on `/data`, `/v1/speak` misses fetched from the upstream server's `/v1/speak` (stdlib urllib) and saved (`X-Source: upstream`), no Gemini key. `scripts/make_edge_env.py` writes `.env.edge`; `ENV_FILE=.env.edge scripts/docker-run.sh edge` runs it. `/v1/speak` misses are single-flight per (text, lang, voice) in both modes. The upstream can run **stateless** with `SPEAK_STORE=0`: it never writes the bank, so clips live only on edges.
- Two halves are decoupled: serving needs **no ML stack** (base requirements + ffmpeg).

## Dev setup & commands
- Python venv at **`.venv`** (Python 3.14). Always invoke via `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/uvicorn`.
- **Building the venv from scratch** (a fresh clone has none, and these each cost time to rediscover):
  - Use **`uv venv .venv --python 3.14`** — stock `python3 -m venv` fails on Debian/Ubuntu without `python3.14-venv` (no `ensurepip`).
  - `src/audio.py` shells out to **ffmpeg**; `apt install ffmpeg`, or for a venv-local fallback `pip install imageio-ffmpeg` and symlink its binary into `.venv/bin/ffmpeg`.
  - Serving + the test suite need only `requirements.txt` (`uv pip install -r requirements.txt`).
- **Tests:** `PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest` (unit — fast, no cloud).
  The `PATH` prefix is **required**, not optional: `src/audio.py` shells out to `ffmpeg`,
  which here is only a symlink inside `.venv/bin` (see the ffmpeg note below). Without it
  7 tests fail with `FileNotFoundError: 'ffmpeg'` — a broken environment that looks like
  broken code. Clean baseline: **217 passed, 2 skipped** in ~13s (the skips are the
  real-bank check without a local bank, and the live test below).
  Keep tests lean: one test per behaviour, no cartesian parametrize grids (a 155-case
  `lang` grid was cut to 21 tests with the same mutation coverage).
  `.venv/bin/pytest --integration tests/test_integration.py` runs `/v1/speak` against
  **real Gemini** (paid, one synthesis): source `.env` first; it needs DNS for
  `generativelanguage.googleapis.com` (blocked on this machine — run in docker with
  `--dns 8.8.8.8`). Without a key it skips; with unresolvable DNS it fails saying so.
- **Run server + UI:** `PATH="$PWD/.venv/bin:$PATH" SERVE_FRONTEND=1 .venv/bin/uvicorn src.server:app --port 8000` → http://localhost:8000. The PATH prefix matters for the same ffmpeg reason as the tests: `/v1/speak` misses encode OGG server-side, and without it they 503 with `ffmpeg not installed`.
- **Creds:** `GEMINI_API_KEY` for word-bank generation/cleanup, `/v1/speak` misses, and `/v1/gemini/tts`; serving bank hits needs none. `JWT_SECRET` (optional) — when set, `/v1/speak` and `/v1/gemini/tts` require a bearer token (unset = open, dev mode); generate it with `scripts/make_jwt_secret.py` (writes `.env`, never prints it; `--rotate` to replace). Secrets live in `.env` (gitignored, `0600`) — never print or cat it.
- **After editing `src/server.py` or `wordbank/lookup.py`, run `.venv/bin/pytest`** — it includes `tests/test_serves_real_bank.py`, which serves a real word from the local bank. Synthetic unit tests alone have repeatedly missed "server returns no audio" regressions (path resolution, WAL read visibility, env defaults). That test skips if no local bank, so also sanity-check by hand against the real data when you have it.

## Data & state (important)
`wordbank.db`, `output/`, `data/`, `dist/` are **gitignored** → a fresh clone has **no bank**. Either regenerate (`fetch_wordlist` → `batch-run` → `encode_wavs`) or pull the bank from storage. The server serves whatever `BANK_DB` / `OGG_ROOT` (and dev-only `WAV_ROOT`) point at. `wordbank.db` holds two tables: `words` (Gemini single-word bank, UNIQUE(word,lang)) and `phrases` (`/v1/speak` phrase bank, UNIQUE(text,lang,voice) — separate so words/phrases never overlap; `Manifest` rebuilds a pre-voice `phrases` table, whose key was UNIQUE(text,lang), once on open). Older banks may also hold an unused `voices` table (the removed clone-reference registry). Both `words` and `phrases` carry a `voice` column (surfaced in lookups); `phrases` also carry legacy `seed`/`ref_audio`/`ref_text` provenance from the removed local pipeline. `Manifest(table=, text_col=)` targets one content table. Backfill `voice` on an old bank with `scripts/backfill_voice.py`.

## Invariants — don't regress these (each cost real debugging)
- **Bank lookup uses a normal SQLite connection + `PRAGMA query_only`, NEVER `mode=ro`.** A read-only URI connection on a WAL DB uses a private in-memory `-shm` and won't see another process's commits → stale/empty results. (`wordbank/lookup.py`)
- **`src/server.py` imports google-genai and `src.audio` lazily, inside handlers — not at top level**, and the image ships `requirements-base.txt` only (no numpy/soundfile). Don't hoist them. Guarded by `tests/test_lazy_imports.py`.
- **Manifest `file_path` is relative to the bank root** (`en/…/<id>_<word>.wav`), resolved at serve time under `WAV_ROOT`/`OGG_ROOT`. Never store absolute paths; changing the layout requires a migration (`scripts/migrate_manifest_paths.py`).
- **Frontend uses same-origin relative URLs** → it must be served by the API (`SERVE_FRONTEND=1`), not a standalone `http.server`.
- **OGG (16k VoIP) is the served format; WAV is a local master only** — never shipped, imaged, or committed.
- **`lang` is the serving code (`en-us`)**; lookups are language-scoped, backed by the `(lang, word)` index. (The on-disk dir is historically `en`; `file_path` is opaque to the server beyond root-joining — fine.)
- **The server is read-only w.r.t. the bank**, with exactly one sanctioned writer: `POST /v1/speak` (`phrases` table, get-or-generate on a miss; disabled by `SPEAK_STORE=0`, the stateless upstream — misses are then coalesced in flight instead of single-flighted via the bank) — via `Manifest` + `bank.close()` for read visibility. Everything else, including `wordbank` generation tools, stays read-only from the server's perspective.
- **Tests fake Gemini by patching `wordbank.gemini_tts.synthesize_pcm`** (`fake_gemini` fixture; the handlers import it lazily, so the patch takes effect). `bank_env` points `BANK_DB`/`OGG_ROOT`/`WAV_ROOT` at a tmp bank — never let a test touch the real one. The `client` fixture combines both and sets a dummy `GEMINI_API_KEY` (a Gemini deployment, so `/health` is green).
- **Every google-genai client comes from `gemini_tts.new_client()`, pinned to `v1beta`.** TTS (`speechConfig`), batches and file uploads are v1beta-only; google-genai 3.x defaults to `v1`, where every synth 400s with `Unknown name "speechConfig"`. Never construct `genai.Client()` directly. Guarded by `tests/test_gemini_tts.py`.
- **Delivery directions go through the capability registry (`gemini_tts.TTS_MODELS` / `route_direction`).** The 3.8 models read any inline direction aloud, so they only ever get one as the `speech_metadata` style, text verbatim; 3.1/2.5 keep the inline prefix and reject a style (raised before any request). `synthesize_pcm`, `GeminiBatch`, the CLI and the server all route through it — never build `f"{instruction} {word}"` for a model yourself. Guarded by `tests/test_gemini_style.py`.
- **`lang` is validated before it touches anything.** Every route taking `lang` runs `_lang()` (`wordbank.languages.normalize_lang`: trimmed, lowercased, `[a-z]{2,3}(-[a-z0-9]{2,8})*`, ≤35 chars, ASCII only) and 400s otherwise, before any lookup, synthesis, upstream fetch or write; the CLI `--lang` uses the same function. It is the first directory of every saved clip — a raw `lang` once wrote outside the bank roots (`../../x`, absolute paths). The writer (`_save_phrase`) also checks containment with `paths.path_under` (realpath) before creating a row or file. Guarded by `tests/test_lang_validation.py`.
- **Gemini audio payloads always go through `gemini_tts.pcm_from_payload`.** `gemini-3.8-flash-lite-tts` returns a full WAV (`fmt ` + `data` + a ~7.6 KB `C2PA` provenance chunk), not raw PCM like 3.1; read as PCM, that is a click plus ~0.15 s of loud noise on every clip. Anything other than 24 kHz / 16-bit / mono raises `GeminiAudioFormatError` (502 from the server) — never resampled. Any new code that reads `inline_data.data` must use it. Guarded by `tests/test_gemini_payload.py`.
- **`wordbank verify` resolves `db` / `ogg_root` independently.** With `--root` it derives the flat deploy layout (`/data/wordbank.db` + `/data/words_ogg`); with no `--root` it falls back to the server's own env defaults, so a bare `python -m wordbank verify --desc library.json` checks exactly what the server would serve. Don't collapse these back to a single root — the dev layout (db at repo root, audio under `output/`) can't be expressed by one.

## Gemini batch notes (generation)
- Enqueued-token quota counts **active (non-terminal)** jobs; it frees on completion. Use **`batch-run`** (quota-aware waves) for big runs; auto-delete of jobs is **off** (cleanup is on-request).
- Not every TTS model supports `batchGenerateContent` — `gemini-3.1-flash-tts-preview` and `2.5-pro-tts` do, `2.5-flash-tts` does not. Confirm with `scripts/list_models.py`.
- **Sync TTS (`/v1/speak`, studio, scripts) uses `gemini_tts.DEFAULT_MODEL` = `gemini-3.8-flash-tts`** (with a `style` it said the words 3.8 Lite acted out; 1.5x Lite's price; `GEMINI_TTS_MODEL` overrides per deployment). The en-us word bank stays on `gemini-3.1-flash-tts-preview` (`BATCH_DEFAULT_MODEL`), so spoken phrases and banked words come from different models. Don't go back to a 2.5 model: both 2.5 TTS models reject imperative text with 400 `Model tried to generate text` and drop audio on digit runs (3/6 vs 6/6 on the same texts).
- **The 3.8 models speak inline instructions aloud.** Prepending a style instruction (`VOICE_INSTRUCTION`, `instruct`) turned a 1 s word into a ~9 s clip of the prompt being read. They take directions only as a **style** (next section); the batch/sync CLIs refuse a 3.8 model with neither `--style` nor an explicit `--instruction` (`--instruction=` = bare word) instead of defaulting to `VOICE_INSTRUCTION`.
- **3.8 voices: 1,000 prebuilt (`GET /v1beta/voices`), 117 of them `en-GB`**, tagged by city accent (Winchester, Bristol, Manchester, Liverpool, Newcastle, Glasgow; none labelled RP), ids like `en-gb-advisor-1`. The batch path passes any voice id through; `/v1/speak` and the studio still only accept the 30 classic names in `gemini_tts.VOICES`.
- A job that ends FAILED/CANCELLED/EXPIRED (no results file) no longer blocks collection: its still-submitted words become `failed` with `batch job JOB_STATE_…` in `error`, and the next submit retries them (each submission is one attempt, `max_attempts` 3). PARTIALLY_SUCCEEDED jobs are collected normally.
- Uploaded batch input file must reach `ACTIVE` before `batches.create` (handled in `submit_file`).
- Cost ≈ **$0.35 / 1000 words** (3.1-flash-tts batch: $10/1M audio-output tokens, 25 tokens/sec). Actual spend is in AI Studio → Usage / Cloud Billing under **"Generative Language API"**, not queryable via the API.
- Adding a language to `/v1/languages`' `generate` list (`wordbank/languages.py`) is a **manual, verified gate** — only after listening to samples.

## Gemini style (3.8 models)
Bare single words on 3.8 sometimes get acted out ("haha"/"laughing" → a laugh track, "russian" → Russian speech). A delivery direction fixes it only if it stays out of the spoken text: the **style**.

| Model | `style` (speech metadata) | inline prefix |
|---|---|---|
| `gemini-3.8-flash-tts` | yes (verified, classic + 3.8-library voices) | **spoken aloud** — never sent |
| `gemini-3.8-flash-lite-tts` | yes (verified with `en-gb-tutor-11`) | **spoken aloud** — never sent |
| `gemini-3.1-flash-tts-preview` | no — 400 `Speech metadata is not supported for this model.` | yes |
| `gemini-2.5-*-preview-tts` | not probed, treated as no | yes |

- Sync: `POST /v1beta/interactions` (`x-goog-api-key`), body `{"model", "input": [{"type": "user_input", "content": [{"type": "text", "text": <verbatim>, "annotations": [{"type": "speech_metadata", "style": <direction>}]}]}], "response_format": {"type": "audio"}, "generation_config": {"speech_config": [{"voice": <voice>}]}}`. Audio is base64 at `steps[type=model_output].content[type=audio].data`, `audio/wav` (24 kHz mono 16-bit, through `pcm_from_payload`). The style text was not counted as input tokens (3 for "laughing" + a 20-word style).
- `system_instruction` (generateContent) is rejected by both 3.8 models (400 `Developer instruction is not enabled for this model`) — don't use it.
- Batch: the Batch API takes generateContent requests only ("This feature is currently only available with the generateContent API."), both 3.8 model pages list Batch API as supported, and generateContent's `Part` has `speechMetadata.style` ("Attach `speech_metadata` … to each `part`"). So `batch*` sends `{"text": <word>, "speechMetadata": {"style": …}}` per part — **documented, not yet tried live**: run `scripts/gemini_batch_smoke.py --model gemini-3.8-flash-tts --style "…"` (8 words) and listen before a bulk run.
- Rate limit: Tier 1 is 10 requests/min per model; `gemini-run --rpm` (default 10) spaces requests 60/rpm s apart, empty-audio retries included.
- Generate a bank with a style:
  ```bash
  WB="python -m wordbank --db wordbank.db --out output/words --lang en-gb"
  $WB batch-run --words words.txt --model gemini-3.8-flash-lite-tts --voice en-gb-tutor-11 --style "…"   # async, half price
  $WB gemini-run --words words.txt --model gemini-3.8-flash-lite-tts --voice en-gb-tutor-11 --style "…"  # sync, verified path, 10 rpm
  ```
- Redo individual words: `$WB regenerate --words-file redo.txt --ogg-root output/words_ogg` moves each word's WAV/OGG to `<db dir>/takes/<UTC>/{wav,ogg}/` (+ `words.json`) and resets the rows to pending; then `$WB gemini-run --model … --voice … --style "…"` (no `--words`: just the pending rows) and `scripts/encode_wavs.py`. Moved, not copied: `encode_wavs.py` skips existing OGGs.

## Key files
- `src/server.py` API · `src/audio.py` PCM→OGG · `src/upstream.py` edge client · `src/auth.py` JWT
- `wordbank/`: `manifest.py` (SQLite — `words` + `phrases` content tables, generic over `table`/`text_col`) · `gemini_batch.py` (Gemini batch → words) · `gemini_run.py` (sync, rate-limited Gemini → words; `gemini-run`) · `gemini_tts.py` (sync Gemini TTS for studio/`/v1/speak`; `new_client`, `pcm_from_payload`, the `TTS_MODELS` capability registry + `route_direction`, the Interactions style path) · `ratelimit.py` (token bucket, injectable clock) · `library.py` (deploy `describe`/`verify`) · `lookup.py` (read-only merged search + voice/type) · `paths.py` (slugs, `path_under`) · `languages.py` · `wordlist.py` · `config.py` · `__main__.py` (CLI: `describe` / `verify` / `batch*` / `gemini-run` / `regenerate` / `status` / `export`)
- `scripts/`: `fetch_wordlist` · `encode_wavs` · `package_bank` · `pack.sh`/`unpack.sh` (one-tarball bundle: code+db+OGG) · `backfill_voice` · `gemini_batch_cleanup` · `gemini_batch_smoke` · `list_models` · `migrate_manifest_paths`
- `frontend/index.html` (bank lookup + Speak, Gemini voice preview) · `Dockerfile` · `docker-compose.yml` (profile `bank`)
- Open work is the **TODO** at the bottom of `README.md`.

---

## Continuing development

> **Always ask before starting work.** Even if a task below looks clear, confirm
> the scope and priority with the user first. Do not pick up any item autonomously.

### Outstanding product goals (in rough priority order)

1. **Multi-language support** — generation is data-driven; what's needed is a
   word list + a Gemini batch run per language, then a verified listen before
   adding the locale to `wordbank/languages.py`.
2. **JWT auth** — **done.** Chosen over API-key (the review's suggestion and
   this TODO were the same layer — one decision covers both). HS256 via
   `JWT_SECRET`, checked by a FastAPI dependency on `/v1/speak` and
   `/v1/gemini/tts`; read-only bank endpoints stay public. Env-gated:
   `JWT_SECRET` unset = open/dev, set = required. Optional `JWT_AUDIENCE`/
   `JWT_ISSUER` add claim checks. Generate the secret into `.env` with
   `scripts/make_jwt_secret.py`; mint tokens with `scripts/make_token.py
   --sub <name> --ttl 3600`.

### Code-quality backlog

The review's dead-code, bloat, and test-quality items are **done** (see git log:
dead code removal → naming/constants → verify fix → thread-safety → tests →
deployment hardening → JWT auth → removal of local OmniVoice synthesis).

**Deliberately NOT done (decided against, don't "fix" these):**
- **Shortening `VOICE_INSTRUCTION`** (`wordbank/gemini_batch.py`) to save ~$1.80
  per 50k-word bank. That string shapes delivery style, and the existing 47,949-word
  bank was generated with it — changing it makes new words subtly inconsistent
  with words already banked, with nothing in the manifest recording which prompt
  produced which clip. Not worth it. Exposing it via `WORDBANK_VOICE_INSTRUCTION`
  as config is fine; changing the default is not.
- **Pre-serializing the static `generationConfig`** in `build_request_body`. It
  trades readability for ~2000 short-lived dict allocations in a job that spends
  minutes blocked on a network API.
- **Removing `BANK_SYNC_CMD`** as a "shell-injection vector". Whoever sets that
  env var already controls the container's command line — it is not a privilege
  boundary. `SYNC_BUCKET`/`SYNC_PREFIX` were added as a structured alternative;
  the escape hatch stays for rclone/gsutil/multi-step pulls.

---

## Runbook: package the bank for upload

Trigger: *"package the audio"*, *"prepare the bank for upload"*, *"stage the audio"*.

Prepares an upload-ready folder; **never uploads**. Prereq: WAV bank exists (`output/words/**.wav`) — else generate first.
```bash
python scripts/package_bank.py            # → dist/bank/  (words_ogg/ + wordbank.db)
```
- Verify the report: if it warns `N OGG < ready words`, the bank is incomplete — tell the user, don't treat it as ready.
- `--bitrate` / `--application` change quality (default `16k voip`); `--force` re-encodes; `--out-dir` stages elsewhere.
- The user uploads `dist/bank/` themselves (the script prints S3/GCS/rclone examples). **Do NOT upload unless asked**; never use destructive flags or make a bucket public.
- Deploy: the container pulls `words_ogg/` + `wordbank.db` into its volume via `SYNC_BUCKET`/`SYNC_PREFIX` (structured, preferred) or `BANK_SYNC_CMD` (arbitrary shell — trusted config only).

## Runbook: build the deployment package (description + archive)

Trigger: *"make an archive"*, *"bundle the library"*, *"deployment package"*, *"so I can deploy it elsewhere"*.

The image is **code only** — it never stores the bank. The **deployment package**
is: the Docker image + `library.json` (description) + `tts-bank.tar.gz` (archive).
`scripts/pack.sh` builds the latter two:
```bash
bash scripts/pack.sh          # → library.json + tts-bank.tar.gz (both gitignored)
```
- `library.json` (`wordbank describe`): languages, word/phrase counts, ogg
  count, and the `folders` a provisioned volume must have. Also bundled inside the archive.
- `tts-bank.tar.gz`: code + `wordbank.db` + served OGG (`output/words_ogg`).
  Excludes the multi-GB WAV masters (`output/words`).

**Docker deploy — two `--target`s, no compose in production** (compose stays
dev-only convenience: `docker compose --profile bank up --build`):
- `--target runtime` (default) — code only, bank comes from a mounted `/data`
  volume (e.g. a k8s PVC) or `SYNC_BUCKET`/`BANK_SYNC_CMD` at boot.
- `--target runtime-fat` — same image, `library.json` + `tts-bank.tar.gz`
  baked in at `/data` at build time (`COPY` in the Dockerfile). No mount
  needed; `docker-entrypoint.sh`'s verify/unpack runs unchanged, it just
  finds the archive already there. For deploying before persistent storage
  exists — every rollout re-bakes the bank into the image. Requires
  `library.json`/`tts-bank.tar.gz` present at build time (`scripts/pack.sh`).

`scripts/docker-build-thin.sh` / `scripts/docker-build-fat.sh` wrap the plain
`docker build --target runtime` / `--target runtime-fat`. `scripts/docker-run.sh thin|fat`
wraps `docker run`: `/data` is a named volume (`whizz-tts-<mode>-data`,
`DATA_VOLUME` overrides) so `/v1/speak` phrases survive restarts; `thin` bind-mounts
local `library.json`/`tts-bank.tar.gz` onto `/data` when both exist, else runs
bank-less (needs `GEMINI_API_KEY`); `edge` runs the thin
image as an edge cache (`SPEAK_UPSTREAM_*` + `JWT_SECRET`, never the Gemini
key; bank files mounted like `thin`). `GEMINI_API_KEY`/`GOOGLE_API_KEY`/
`JWT_SECRET` pass through by name (`ENV_FILE`, default `./.env`, loaded first;
`HOST_PORT` sets the published port);
`DOCKER_DNS=8.8.8.8` sets the container resolver, for a local DNS that won't
resolve `generativelanguage.googleapis.com`. Tags: `whizz-tts:thin`,
`whizz-tts:fat`.

Put `library.json` + `tts-bank.tar.gz` on the (persistent)
`/data` volume; `docker-entrypoint.sh` on boot:
0. If `SYNC_BUCKET`/`SYNC_PREFIX` (or `BANK_SYNC_CMD`) is set, pulls the bank
   into the volume. **This runs first** — it is a way of *provisioning* the
   volume, so it has to happen before the check below decides the volume is
   unusable. Don't reorder it after the verify.
1. `python -m wordbank verify --desc $LIBRARY_DESC --root $DATA_DIR` (defaults
   `/data/library.json`, `/data`) — checks the described folders exist, the DB
   word count matches, and phrase + ogg counts are **at least** the described ones
   (`/v1/speak` grows both on the live volume; exact matching made every restart
   after a speak write re-unpack the archive over the grown `wordbank.db`).
2. If invalid, unpacks `$LIBRARY_ARCHIVE` (`/data/tts-bank.tar.gz`) into `/data`
   (`output/words_ogg`→`/data/words_ogg`, `wordbank.db`→`/data/wordbank.db`,
   `library.json`; also `output/voices`→`/data/voices` from archives packed
   before clone references were removed, whose description lists that folder),
   then re-verifies. An
   existing `wordbank.db` (+ `-wal`/`-shm`), e.g. phrases `/v1/speak` saved
   before the archive was added, is first moved to
   `wordbank.db.pre-unpack-<timestamp>` — never overwritten. A volume with a
   runtime-built bank but no `library.json` starts as-is when synthesis is
   available (logged as such, not as "no bank").
3. **Refuses to start (`exit 1`) if there is still no usable bank** — rather than
   coming up "healthy" and 404-ing every lookup. Exceptions, both valid bank-less
   deployments that degrade to a warning instead: `GEMINI_API_KEY`/
   `GOOGLE_API_KEY` set, or `SPEAK_UPSTREAM_URL` set (edge) — `/v1/speak` then generates or fetches misses and builds the
   phrase bank on the volume. `/health` applies the
   identical rule (503 only when no bank, Gemini key, or upstream can serve), and
   the image `HEALTHCHECK` is wired to it.
4. Starts the server. Later restarts find the volume provisioned → no unpack.

**Bare-metal**: `mkdir tts-bank && tar xzf tts-bank.tar.gz -C tts-bank && cd tts-bank
&& bash scripts/unpack.sh` (`.venv` + `requirements-base.txt` only, verifies counts).
Distinct from `package_bank.py`, which stages data-only for upload to object storage.

## Runbook: free the Gemini batch enqueued-token quota

Trigger: *"delete the batch jobs"*, *"free the enqueued-token quota"*, *"clean up gemini batches"*.

The quota counts **active (non-terminal)** jobs and frees on completion. Lingering enqueued tokens with "nothing running" almost always means **jobs stuck in `PENDING`/`RUNNING`**. The pipeline does **not** auto-delete, so do this **only when asked**. Needs `GEMINI_API_KEY`.
```bash
python scripts/gemini_batch_cleanup.py                     # list jobs + states (no changes)
python scripts/gemini_batch_cleanup.py --delete --cancel   # cancel stuck + delete finished
```
- **List first.** Non-terminal jobs are the real usage → `--cancel` stops them. Terminal jobs don't count; `--delete` is tidy-up only.
- Safe (audio already collected); only removes job records. Restricted to `wordbank-` jobs unless `--all`. Don't delete jobs whose results weren't collected.
