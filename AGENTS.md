# Agent guide

Orientation + runbooks for common tasks. See `README.md` and `WORDBANK_PLAN.md`
for full detail.

## Project at a glance
- **Generate** (`wordbank/`, Gemini Batch API) → WAV masters in `output/words/` + manifest `wordbank.db`.
- **Encode** (`scripts/encode_wavs.py`) → OGG copies in `output/words_ogg/` (16k VoIP, ~3 KB/word).
- **Serve** (`src/server.py`, FastAPI) → bank lookup over the manifest + static OGG; live OmniVoice synth optional behind `MODEL_ENABLED`.
- Manifest `file_path` is **relative to the bank root**; `lang` is the serving code (e.g. `en-us`).

---

## Runbook: package the bank for upload

Trigger phrases: *"package the audio"*, *"prepare the bank for upload"*, *"stage the audio for storage"*.

The script **prepares** an upload-ready folder; it never uploads. The user
uploads it wherever they want themselves.

**Prerequisites**
- WAV bank exists (`output/words/**.wav`). If not, generate first (README → *Generate / maintain the bank*).

**Do it** — encodes (WAV→OGG 16k VoIP), checkpoints the manifest, and stages everything into one folder:
```bash
python scripts/package_bank.py            # → dist/bank/  (words_ogg/ + wordbank.db)
```

**Verify**
- The script reports `staged N OGG files`. If it warns `N OGG < ready words`, the bank is incomplete — tell the user; don't treat it as ready.
- Output is `dist/bank/` containing `words_ogg/` (hardlinked, no extra disk) + `wordbank.db`. **WAV masters stay local.**

**Options**
- Quality/size: `--bitrate` / `--application` (default `16k voip` ≈ 3 KB/word; e.g. `--bitrate 32k --application audio`).
- `--force` re-encodes existing OGG copies (use after changing bitrate).
- `--out-dir DIR` to stage elsewhere.

**Then the user uploads** `dist/bank/` to any storage (S3/GCS/rclone/rsync — the script prints example commands). **Do NOT upload for them unless asked**, and never use destructive flags (`aws s3 rm --recursive`, `sync --delete`) or make a bucket public.

**Deploy** — the container pulls `words_ogg/` + `wordbank.db` into its volume at startup via `BANK_SYNC_CMD` (see `docker-compose.yml`), then serves OGG with no ML stack.

---

## Runbook: free the Gemini batch enqueued-token quota

Trigger phrases: *"delete the batch jobs"*, *"free the enqueued-token quota"*, *"clean up gemini batches"*.

The enqueued-token quota counts **active (non-terminal) batch jobs**, so it frees
when a job completes. If you still see enqueued tokens with "nothing running," the
culprit is almost always **jobs stuck in `PENDING`/`RUNNING`** — list to find them.

The pipeline does **NOT** auto-delete jobs (`delete_after` defaults off), so do
this **only when the user asks** to clean up. Needs `GEMINI_API_KEY`.

```bash
python scripts/gemini_batch_cleanup.py            # list jobs + states (no changes)
python scripts/gemini_batch_cleanup.py --delete --cancel   # cancel stuck + delete finished
```
- **List first.** Non-terminal (`PENDING`/`RUNNING`) jobs are the real enqueued usage → `--cancel` stops them. Terminal jobs don't count toward the quota; `--delete` is just tidy-up (and frees server-side storage / avoids the 48h pile-up).
- Safe because the bank's audio is already collected; this only removes job records. Restricted to `wordbank-` display names unless `--all`.
- Do NOT delete jobs whose results the user hasn't collected.
