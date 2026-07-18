# Agent guide

Orientation, dev setup, invariants, and runbooks. Full detail in `README.md`
and `WORDBANK_PLAN.md`.

## Project at a glance
- **Generate** → WAV masters in `output/words/` + manifest `wordbank.db`, via either the **Gemini Batch** path (`gemini_batch.py` → the `words` table) or the **local OmniVoice** path (`wordbank generate` → the **separate `phrases` table**, cloning a Gemini reference). Words and phrases are stored in distinct tables so they never overlap. Each local phrase records the **RNG seed + clone reference** used (`seed`/`ref_audio`/`ref_text`) → reproducible.
- **Encode** (`scripts/encode_wavs.py`) → OGG copies in `output/words_ogg/` (16k VoIP, ~3 KB/word).
- **Serve** (`src/server.py`, FastAPI) → indexed bank lookup over the manifest + static OGG; live OmniVoice synth optional behind `MODEL_ENABLED`. `GET /v1/search` is the merged word+phrase lookup — each result carries a `type` flag (`word`|`phrase`) and the `voice` it was made with; phrases serve from `/v1/phrase`. Also hosts the **voice studio**: `GET /v1/gemini/voices` (the 30 prebuilt Gemini voices), `POST /v1/gemini/tts` (preview a Gemini phrase), `POST/GET /v1/voices` (save + list clone references), and `/v1/tts` accepts `voice=<saved name>` to clone from a stored clip. Gemini voices are **style presets, not language-bound** — language is auto-detected from the text — so the studio's flag dropdown is the (filing-only) bank language, not the voice.
- Two halves are decoupled: serving needs **no ML stack** unless live synth is on.

## Dev setup & commands
- Python venv at **`.venv`** (Python 3.14). Always invoke via `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/uvicorn`.
- **Building the venv from scratch** (a fresh clone has none, and these each cost time to rediscover):
  - Use **`uv venv .venv --python 3.14`** — stock `python3 -m venv` fails on Debian/Ubuntu without `python3.14-venv` (no `ensurepip`).
  - Install **`torch` and `torchaudio` from the same index**: `uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu`. Mixing the CPU index with default PyPI gives an ABI mismatch that surfaces as `OSError: Could not load .../_torchaudio.abi3.so`.
  - `src/audio.py` shells out to **ffmpeg**; `apt install ffmpeg`, or for a venv-local fallback `pip install imageio-ffmpeg` and symlink its binary into `.venv/bin/ffmpeg`.
  - Serving + the fast test suite need only: `fastapi uvicorn[standard] python-multipart pytest numpy soundfile httpx librosa`. `torch`/`omnivoice` are needed only for model tests and live synth.
- **Tests:** `.venv/bin/pytest` (unit — fast, no model/cloud). `.venv/bin/pytest --integration tests/test_integration.py` loads OmniVoice (slow, needs the model).
- **Run server + UI (bank only):** `SERVE_FRONTEND=1 .venv/bin/uvicorn src.server:app --port 8000` → http://localhost:8000
- **Creds:** `GEMINI_API_KEY` for word-bank generation/cleanup **and the voice-studio endpoints** (`/v1/gemini/tts`, `gemini-voice`); bank serving needs none. Secrets live in `.env` (gitignored).
- **After editing `src/server.py` or `wordbank/lookup.py`, run `.venv/bin/pytest`** — it includes `tests/test_serves_real_bank.py`, which serves a real word from the local bank. Synthetic unit tests alone have repeatedly missed "server returns no audio" regressions (path resolution, WAL read visibility, env defaults). That test skips if no local bank, so also sanity-check by hand against the real data when you have it.

## Data & state (important)
`wordbank.db`, `output/`, `data/`, `dist/` are **gitignored** → a fresh clone has **no bank**. Either regenerate (`fetch_wordlist` → `batch-run` → `encode_wavs`) or pull the bank from storage. The server serves whatever `BANK_DB` / `OGG_ROOT` (and dev-only `WAV_ROOT`) point at. `wordbank.db` holds three tables: `words` (Gemini single-word bank, UNIQUE(word,lang)), `phrases` (local OmniVoice phrase bank, UNIQUE(text,lang,voice) — separate so words/phrases never overlap), and `voices` (clone references — generation-side only, never served). Both `words` and `phrases` carry a `voice` column (surfaced in lookups); `phrases` also carry `seed`/`ref_audio`/`ref_text` (provenance — the local pipeline records them so every clip is reproducible). Reference clips live under `VOICES_ROOT` (default `output/voices`, gitignored). `Manifest(table=, text_col=)` targets one content table. Backfill `voice` on an old bank with `scripts/backfill_voice.py`.

## Invariants — don't regress these (each cost real debugging)
- **Bank lookup uses a normal SQLite connection + `PRAGMA query_only`, NEVER `mode=ro`.** A read-only URI connection on a WAL DB uses a private in-memory `-shm` and won't see another process's commits → stale/empty results. (`wordbank/lookup.py`)
- **`src/server.py` imports `src.model` / `src.audio` lazily, inside handlers — not at top level.** That's what lets the bank-only image ship without torch/omnivoice. Don't hoist them.
- **Manifest `file_path` is relative to the bank root** (`en/…/<id>_<word>.wav`), resolved at serve time under `WAV_ROOT`/`OGG_ROOT`. Never store absolute paths; changing the layout requires a migration (`scripts/migrate_manifest_paths.py`).
- **Frontend uses same-origin relative URLs** → it must be served by the API (`SERVE_FRONTEND=1`), not a standalone `http.server`.
- **OGG (16k VoIP) is the served format; WAV is a local master only** — never shipped, imaged, or committed.
- **`lang` is the serving code (`en-us`)**; lookups are language-scoped, backed by the `(lang, word)` index. (The on-disk dir is historically `en`; `file_path` is opaque to the server beyond root-joining — fine.)
- **The server is read-only w.r.t. the bank.** Only `wordbank` generation tools write `wordbank.db`.
- **Tests mock the model on `src.model`** (the handler imports it lazily, so patching `src.server` won't take). The `client` fixture forces `MODEL_ENABLED=1` and patches `src.model.generate`/`ensure_loaded`; bank tests set `BANK_DB`/`OGG_ROOT`/`WAV_ROOT` via env. Keep this pattern when editing the server.
- **All access to the OmniVoice model goes through `_MODEL_LOCK` (`src/model.py`).** The model isn't thread-safe and `/v1/tts` offloads synthesis to a multi-threaded executor. The lock covers **seeding *and* generation as one unit** — `_seed_everything` sets torch's *global* RNG, so splitting them lets one request silently generate with another's seed, breaking the reproducibility guarantee the phrase bank records seeds for. It's an `RLock` because `ensure_loaded` → `load_model` nests. Guarded by `tests/test_model_concurrency.py`.
- **`wordbank verify` resolves `db` / `ogg_root` / `voices_root` independently.** With `--root` it derives the flat deploy layout (`/data/wordbank.db` + `/data/words_ogg`); with no `--root` it falls back to the server's own env defaults, so a bare `python -m wordbank verify --desc library.json` checks exactly what the server would serve. Don't collapse these back to a single root — the dev layout (db at repo root, audio under `output/`) can't be expressed by one.

## Gemini batch notes (generation)
- Enqueued-token quota counts **active (non-terminal)** jobs; it frees on completion. Use **`batch-run`** (quota-aware waves) for big runs; auto-delete of jobs is **off** (cleanup is on-request).
- Not every TTS model supports `batchGenerateContent` — `gemini-3.1-flash-tts-preview` and `2.5-pro-tts` do, `2.5-flash-tts` does not. Confirm with `scripts/list_models.py`.
- Uploaded batch input file must reach `ACTIVE` before `batches.create` (handled in `submit_file`).
- Cost ≈ **$0.35 / 1000 words** (3.1-flash-tts batch: $10/1M audio-output tokens, 25 tokens/sec). Actual spend is in AI Studio → Usage / Cloud Billing under **"Generative Language API"**, not queryable via the API.
- Adding a language to `/v1/languages`' `generate` list (`wordbank/languages.py`) is a **manual, verified gate** — only after listening to samples.

## Key files
- `src/server.py` API · `src/model.py` OmniVoice (lazy) · `src/audio.py` PCM→OGG
- `wordbank/`: `manifest.py` (SQLite — `words` + `phrases` content tables (generic over `table`/`text_col`) + `voices` registry) · `gemini_batch.py` (Gemini batch → words) · `gemini_tts.py` (sync Gemini TTS for studio/refs) · `voices.py` (clip storage + path resolution) · `library.py` (deploy `describe`/`verify`) · `pipeline.py` + `backends/local.py` (local OmniVoice sync gen → phrases) · `lookup.py` (read-only merged search + voice/type) · `languages.py` · `config.py` · `__main__.py` (CLI: `generate` (phrases) / `gemini-voice` / `add-voice` / `voices` / `describe` / `verify` / `batch*` / `status` / `export`)
- `scripts/`: `fetch_wordlist` · `make_accent_reference` · `encode_wavs` · `package_bank` · `pack.sh`/`unpack.sh` (one-tarball bundle: code+db+OGG) · `backfill_voice` · `gemini_batch_cleanup` · `gemini_batch_smoke` · `list_models` · `migrate_manifest_paths`
- `frontend/index.html` also has the **Gemini voice studio** (author refs) + a saved-reference picker in the cloning panel.
- `frontend/index.html` · `Dockerfile` (`--build-arg WITH_MODEL`) · `docker-compose.yml` (profiles `bank` / `full`)
- Open work is the **TODO** at the bottom of `README.md` (verify voice cloning, en-GB bank, multi-language, JWT auth).

---

## Continuing development

> **Always ask before starting work.** Even if a task below looks clear, confirm
> the scope and priority with the user first. Do not pick up any item autonomously.

### Outstanding product goals (in rough priority order)

1. **Validate voice-cloning quality** — the en-GB phrase pipeline is wired but
   the output hasn't been listened to. Generate a small set (`--lang en-gb
   generate`) and present samples for review before expanding.
2. **Fill the phrase bank** — author reference voices per accent (Gemini voice
   studio → `add-voice`), then generate phrase sets per locale. Serving and
   merged search (`/v1/search`) are already ready.
3. **Multi-language support** — generation is data-driven; what's needed is a
   word list + a reference voice per language, then a verified listen before
   adding the locale to `wordbank/languages.py`.
4. **JWT auth** — API is open; add bearer-token verification as a FastAPI
   dependency on all write/synthesis routes (`/v1/tts`, `/v1/voices`,
   `/v1/gemini/tts`). Read-only bank endpoints can remain public.

### Code-quality backlog (from `review-report.json`)

The review's dead-code, bloat, and test-quality items are **done** (see git log:
dead code removal → naming/constants → verify fix → thread-safety → tests →
deployment hardening). What remains:

**Still open:**
- **Auth** — the API is open; all write/synthesis routes (`/v1/tts`, `/v1/voices`,
  `/v1/gemini/tts`) are unauthenticated. The review suggested an API-key
  dependency, the TODO below calls for JWT — **they are the same layer, so pick
  one before building it.** `CORS_ORIGINS` narrows browser access but is *not*
  authentication and stops no non-browser client.
- `src/device.py` has no fast-suite coverage. `tests/test_device.py` was deleted
  (all 6 tests asserted only that `unittest.mock` returns what it was told). A
  real check belongs in the `--integration` suite, against actual hardware.
- Model-string duplication: `"gemini-2.5-flash-preview-tts"` is still hardcoded
  in `scripts/gemini_sync.py` and `scripts/make_accent_reference.py`, so those
  can drift from `gemini_tts.DEFAULT_MODEL` independently.

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

## Runbook: accent bank via local OmniVoice + Gemini reference

The local OmniVoice model generates the **phrases**. Accent comes from a one-off
**voice-clone reference** — a phrase generated by Gemini in the target accent,
which OmniVoice clones for the whole bank. No Chirp.

**Reference clip format:** any file `soundfile`/`librosa` can decode — WAV/FLAC/OGG
natively (no ffmpeg), MP3/M4A via ffmpeg. OmniVoice auto-downmixes to mono,
resamples to **24 kHz**, loudness-normalizes (boosts quiet clips), and trims
silence — so you don't pre-process. Keep it **3–10 s**; >20 s warns and degrades
cloning quality (and is auto-trimmed only when `ref_text` is omitted, so a long
clip *with* a transcript stays long — avoid it). `make_accent_reference.py`
already emits the ideal 24 kHz mono WAV.

**Easiest path** (synth + register in one step): the UI **Gemini voice studio**, or
`python -m wordbank --lang <lang> gemini-voice --text "…" --name <name> [--instruct "…"]`.
Then skip to step 3 with `--voice <name>`. Steps 1–2 below are the manual,
file-based equivalent (useful when you already have a clip).

1. **Make the accent reference** (Gemini, one phrase → `ref.wav` + `ref.txt`):
   ```bash
   GEMINI_API_KEY=… python scripts/make_accent_reference.py \
     --text "The quick brown fox jumps over the lazy dog." \
     --instruct "Speak in a British (RP) accent" --voice Kore --out refs/en-gb
   ```
2. **Register the voice** in the `voices` table (so generation can reuse it):
   ```bash
   python -m wordbank --lang en-gb add-voice --ref-audio refs/en-gb/ref.wav \
     --ref-text-file refs/en-gb/ref.txt --source gemini
   ```
3. **Generate locally** — `generate --voice en-gb` pulls the registered reference
   (pass `--ref-audio`/`--ref-text` to override). Verification is off by default
   (Whisper per phrase is slow); add `--verify` to spot-check:
   ```bash
   python -m wordbank --lang en-gb generate --words PHRASES.txt --voice en-gb
   ```
   Sequential (the model isn't thread-safe). Writes WAV masters to
   `output/words/<lang>/local-<voice>/…`; manifest `lang=<lang>`, relative paths.
4. **Encode + package** (`scripts/encode_wavs.py`, `scripts/package_bank.py`).
5. **Listen**, then add the locale to `wordbank/languages.py` `SUPPORTED_LANGUAGES`
   (presence = verified) → it appears in `/v1/languages`.

Serving is already language-scoped (`/v1/bank?lang=en-gb`, `(lang, word)` index) —
no serving change. The Gemini batch path + its en-us word bank are untouched.

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
- `library.json` (`wordbank describe`): languages, voices, word/phrase counts, ogg
  count, and the `folders` a provisioned volume must have. Also bundled inside the archive.
- `tts-bank.tar.gz`: code + `wordbank.db` + served OGG (`output/words_ogg` + `output/voices`).
  Excludes the multi-GB WAV masters (`output/words`).

**Docker deploy** — put `library.json` + `tts-bank.tar.gz` on the (persistent)
`/data` volume; `docker-entrypoint.sh` on boot:
0. If `SYNC_BUCKET`/`SYNC_PREFIX` (or `BANK_SYNC_CMD`) is set, pulls the bank
   into the volume. **This runs first** — it is a way of *provisioning* the
   volume, so it has to happen before the check below decides the volume is
   unusable. Don't reorder it after the verify.
1. `python -m wordbank verify --desc $LIBRARY_DESC --root $DATA_DIR` (defaults
   `/data/library.json`, `/data`) — checks the described folders exist and the
   DB word/phrase counts (+ ogg count) match.
2. If invalid, unpacks `$LIBRARY_ARCHIVE` (`/data/tts-bank.tar.gz`) into `/data`
   (`output/words_ogg`→`/data/words_ogg`, `output/voices`→`/data/voices`,
   `wordbank.db`→`/data/wordbank.db`, `library.json`), then re-verifies.
3. **Refuses to start (`exit 1`) if there is still no usable bank** — rather than
   coming up "healthy" and 404-ing every lookup. Exception: `MODEL_ENABLED=1` is
   a valid bank-less (live-synthesis-only) deployment, so it degrades to a
   warning instead. `/health` applies the identical rule (503 only when neither
   bank nor model can serve), and the image `HEALTHCHECK` is wired to it.
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
