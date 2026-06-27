# Agent guide

Orientation, dev setup, invariants, and runbooks. Full detail in `README.md`
and `WORDBANK_PLAN.md`.

## Project at a glance
- **Generate** (`wordbank/`, Gemini Batch API) → WAV masters in `output/words/` + manifest `wordbank.db`.
- **Encode** (`scripts/encode_wavs.py`) → OGG copies in `output/words_ogg/` (16k VoIP, ~3 KB/word).
- **Serve** (`src/server.py`, FastAPI) → indexed bank lookup over the manifest + static OGG; live OmniVoice synth optional behind `MODEL_ENABLED`.
- Two halves are decoupled: serving needs **no ML stack** unless live synth is on.

## Dev setup & commands
- Python venv at **`.venv`** (Python 3.14). Always invoke via `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/uvicorn`.
- **Tests:** `.venv/bin/pytest` (unit — fast, no model/cloud). `.venv/bin/pytest --integration tests/test_integration.py` loads OmniVoice (slow, needs the model).
- **Run server + UI (bank only):** `SERVE_FRONTEND=1 .venv/bin/uvicorn src.server:app --port 8000` → http://localhost:8000
- **Creds:** `GEMINI_API_KEY` for generation/cleanup; serving needs none. Secrets live in `.env` (gitignored).
- **After editing `src/server.py` or `wordbank/lookup.py`, run `.venv/bin/pytest`** — it includes `tests/test_serves_real_bank.py`, which serves a real word from the local bank. Synthetic unit tests alone have repeatedly missed "server returns no audio" regressions (path resolution, WAL read visibility, env defaults). That test skips if no local bank, so also sanity-check by hand against the real data when you have it.

## Data & state (important)
`wordbank.db`, `output/`, `data/`, `dist/` are **gitignored** → a fresh clone has **no bank**. Either regenerate (`fetch_wordlist` → `batch-run` → `encode_wavs`) or pull the bank from storage. The server serves whatever `BANK_DB` / `OGG_ROOT` (and dev-only `WAV_ROOT`) point at.

## Invariants — don't regress these (each cost real debugging)
- **Bank lookup uses a normal SQLite connection + `PRAGMA query_only`, NEVER `mode=ro`.** A read-only URI connection on a WAL DB uses a private in-memory `-shm` and won't see another process's commits → stale/empty results. (`wordbank/lookup.py`)
- **`src/server.py` imports `src.model` / `src.audio` lazily, inside handlers — not at top level.** That's what lets the bank-only image ship without torch/omnivoice. Don't hoist them.
- **Manifest `file_path` is relative to the bank root** (`en/…/<id>_<word>.wav`), resolved at serve time under `WAV_ROOT`/`OGG_ROOT`. Never store absolute paths; changing the layout requires a migration (`scripts/migrate_manifest_paths.py`).
- **Frontend uses same-origin relative URLs** → it must be served by the API (`SERVE_FRONTEND=1`), not a standalone `http.server`.
- **OGG (16k VoIP) is the served format; WAV is a local master only** — never shipped, imaged, or committed.
- **`lang` is the serving code (`en-us`)**; lookups are language-scoped, backed by the `(lang, word)` index. (The on-disk dir is historically `en`; `file_path` is opaque to the server beyond root-joining — fine.)
- **The server is read-only w.r.t. the bank.** Only `wordbank` generation tools write `wordbank.db`.
- **Tests mock the model on `src.model`** (the handler imports it lazily, so patching `src.server` won't take). The `client` fixture forces `MODEL_ENABLED=1` and patches `src.model.generate`/`ensure_loaded`; bank tests set `BANK_DB`/`OGG_ROOT`/`WAV_ROOT` via env. Keep this pattern when editing the server.

## Gemini batch notes (generation)
- Enqueued-token quota counts **active (non-terminal)** jobs; it frees on completion. Use **`batch-run`** (quota-aware waves) for big runs; auto-delete of jobs is **off** (cleanup is on-request).
- Not every TTS model supports `batchGenerateContent` — `gemini-3.1-flash-tts-preview` and `2.5-pro-tts` do, `2.5-flash-tts` does not. Confirm with `scripts/list_models.py`.
- Uploaded batch input file must reach `ACTIVE` before `batches.create` (handled in `submit_file`).
- Cost ≈ **$0.35 / 1000 words** (3.1-flash-tts batch: $10/1M audio-output tokens, 25 tokens/sec). Actual spend is in AI Studio → Usage / Cloud Billing under **"Generative Language API"**, not queryable via the API.
- Adding a language to `/v1/languages`' `generate` list (`wordbank/languages.py`) is a **manual, verified gate** — only after listening to samples.

## Key files
- `src/server.py` API · `src/model.py` OmniVoice (lazy) · `src/audio.py` PCM→OGG
- `wordbank/`: `manifest.py` (SQLite) · `gemini_batch.py` (batch pipeline) · `lookup.py` (read-only search) · `pipeline.py` (sync Chirp path) · `languages.py` · `config.py` · `__main__.py` (CLI: `batch` / `batch-run` / `batch-submit` / `batch-collect` / `status` / `export`)
- `scripts/`: `fetch_wordlist` · `encode_wavs` · `package_bank` · `gemini_batch_cleanup` · `gemini_batch_smoke` · `list_models` · `migrate_manifest_paths`
- `frontend/index.html` · `Dockerfile` (`--build-arg WITH_MODEL`) · `docker-compose.yml` (profiles `bank` / `full`)
- Open work is the **TODO** at the bottom of `README.md` (verify voice cloning, en-GB bank, multi-language, JWT auth).

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
- Deploy: the container pulls `words_ogg/` + `wordbank.db` into its volume via `BANK_SYNC_CMD`.

## Runbook: free the Gemini batch enqueued-token quota

Trigger: *"delete the batch jobs"*, *"free the enqueued-token quota"*, *"clean up gemini batches"*.

The quota counts **active (non-terminal)** jobs and frees on completion. Lingering enqueued tokens with "nothing running" almost always means **jobs stuck in `PENDING`/`RUNNING`**. The pipeline does **not** auto-delete, so do this **only when asked**. Needs `GEMINI_API_KEY`.
```bash
python scripts/gemini_batch_cleanup.py                     # list jobs + states (no changes)
python scripts/gemini_batch_cleanup.py --delete --cancel   # cancel stuck + delete finished
```
- **List first.** Non-terminal jobs are the real usage → `--cancel` stops them. Terminal jobs don't count; `--delete` is tidy-up only.
- Safe (audio already collected); only removes job records. Restricted to `wordbank-` jobs unless `--all`. Don't delete jobs whose results weren't collected.
