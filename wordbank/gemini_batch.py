"""Gemini Batch API path for bulk word-bank generation (async, ~half cost).

File-based JSONL with an explicit per-request KEY = manifest id, so every result
maps **1:1** to its requested word regardless of response order or omissions —
not positional. At receive time the returned PCM (24 kHz / 16-bit / mono) is
written straight to **WAV** via the stdlib `wave` module: no ffmpeg, no trimming,
no normalization. Any OGG/Opus encoding or trimming is a separate later pass over
the WAVs on disk (see scripts/encode_wavs.py).

Text↔recording index: the manifest `word` column holds the exact requested text
and `file_path` the WAV — `SELECT word, file_path FROM words WHERE status='done'`.

google-genai is lazily imported; tests inject a fake batch client.

⚠️ Confirm with scripts/gemini_batch_smoke.py + a real key before a bulk run:
  - batch accepts the per-request TTS generationConfig on the model;
  - result audio is base64 at candidates[0].content.parts[*].inlineData.data;
  - the leading voice instruction steers delivery and is NOT spoken.
"""
import base64
import json
import os
import pathlib
import tempfile
import time
import wave
from typing import Callable, Iterable, List, Optional

from .constants import SAMPLE_RATE
from .manifest import DONE, FAILED, SUBMITTED, Manifest
from .paths import safe_filename

# NOTE: not every TTS model supports batchGenerateContent. As of this writing
# gemini-3.1-flash-tts-preview and gemini-2.5-pro-preview-tts do; the older
# gemini-2.5-flash-preview-tts does NOT. Run scripts/list_models.py to recheck.
BATCH_DEFAULT_MODEL = "gemini-3.1-flash-tts-preview"
_TERMINAL = ("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED")

# Style direction prepended to each word. Gemini TTS treats a leading "Say …:"
# clause as delivery instruction, not text to speak. Confirm via the smoke test.
VOICE_INSTRUCTION = "Say the following single word clearly and naturally, in a neutral tone:"


# --- request / response helpers (pure, easily testable) --------------------

def build_request_body(word: str, voice: str, instruction: str = VOICE_INSTRUCTION) -> dict:
    """Raw REST GenerateContentRequest body (camelCase) for one word's audio."""
    text = f"{instruction} {word}" if instruction else word
    return {
        "contents": [{"parts": [{"text": text}], "role": "user"}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}
            },
        },
    }


def build_jsonl(rows, voice: str, instruction: str = VOICE_INSTRUCTION) -> bytes:
    """One JSONL line per word: {"key": "<id>", "request": {...}}."""
    lines = [
        json.dumps({"key": str(r["id"]),
                    "request": build_request_body(r["word"], voice, instruction)})
        for r in rows
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def audio_from_line(line: dict) -> Optional[bytes]:
    """Decode base64 PCM from one output JSONL line; None if absent."""
    resp = line.get("response")
    if not resp:
        return None
    try:
        parts = resp["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError):
        return None
    for p in parts:
        blob = p.get("inlineData") or p.get("inline_data")
        if blob and blob.get("data"):
            return base64.b64decode(blob["data"])
    return None


def write_wav_from_pcm_bytes(path: pathlib.Path, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> None:
    """Wrap raw 16-bit mono PCM in a WAV container — no ffmpeg, lossless."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


# --- batch client wrapper (google-genai lazily imported) -------------------

class GeminiBatch:
    def __init__(self, model: str = BATCH_DEFAULT_MODEL, voice: str = "Kore",
                 instruction: str = VOICE_INSTRUCTION, client=None):
        self.model = model
        self.voice = voice
        self.instruction = instruction
        self._client = client

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client()
        return self._client

    def _wait_file_active(self, file, timeout: float = 180, interval: float = 2):
        """Block until an uploaded file is ACTIVE (else batches.create 400s)."""
        client = self._get_client()

        def state_of(f):
            s = getattr(f, "state", None)
            return getattr(s, "name", None) or (str(s) if s is not None else None)

        waited = 0.0
        while True:
            st = state_of(file)
            if st == "ACTIVE":
                return file
            if st == "FAILED":
                raise RuntimeError(f"uploaded batch file {file.name} failed processing")
            if waited >= timeout:
                raise RuntimeError(
                    f"batch file {file.name} not ACTIVE after {timeout}s (state={st})"
                )
            time.sleep(interval)
            waited += interval
            file = client.files.get(name=file.name)

    def submit_file(self, jsonl: bytes, display_name: str) -> str:
        from google.genai import types
        client = self._get_client()
        with tempfile.NamedTemporaryFile("wb", suffix=".jsonl", delete=False) as f:
            f.write(jsonl)
            path = f.name
        try:
            up = client.files.upload(
                file=path,
                config=types.UploadFileConfig(display_name=display_name, mime_type="jsonl"),
            )
            up = self._wait_file_active(up)
            job = client.batches.create(
                model=self.model, src=up.name, config={"display_name": display_name}
            )
        finally:
            os.unlink(path)
        return job.name

    def poll(self, job_name: str):
        return self._get_client().batches.get(name=job_name)

    def is_done(self, job) -> bool:
        state = getattr(job, "state", "")
        name = getattr(state, "name", None) or str(state)
        return name.endswith(_TERMINAL)

    def result_lines(self, job) -> Iterable[dict]:
        content = self._get_client().files.download(file=job.dest.file_name)
        for line in content.decode("utf-8").splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)

    def delete(self, job_name: str) -> None:
        self._get_client().batches.delete(name=job_name)


# --- pipeline --------------------------------------------------------------

def _write_and_mark(manifest: Manifest, out_dir: pathlib.Path, lang: str,
                    voice: str, row, pcm: bytes) -> None:
    # Store the path RELATIVE to the bank root (out_dir) so the manifest is
    # location-independent: the server resolves it under WAV_ROOT/OGG_ROOT,
    # which differ between dev (output/words) and the container (/data/words).
    rel = os.path.join(lang, f"gemini-{voice}", str(row["id"] // 1000),
                       f"{row['id']}_{safe_filename(row['word'])}.wav")
    path = out_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    write_wav_from_pcm_bytes(path, pcm)
    manifest.set_status(row["id"], DONE, file_path=rel,
                        billed_chars=len(row["word"]), voice=voice)


def _collect(manifest: Manifest, batch: GeminiBatch, job_name: str,
             out_dir: pathlib.Path, lang: str, voice: str,
             poll_interval: float, sleep: Callable, log: Callable,
             delete_after: bool = False) -> None:
    while True:
        job = batch.poll(job_name)
        if batch.is_done(job):
            break
        log(f"{job_name}: {getattr(job, 'state', '?')} — waiting {poll_interval}s")
        sleep(poll_interval)

    rows_by_key = {str(r["id"]): r for r in manifest.get_by_batch(job_name)}
    seen = set()
    for line in batch.result_lines(job):
        key = str(line.get("key"))
        row = rows_by_key.get(key)
        if row is None:
            log(f"{job_name}: result for unknown key {key!r} — skipped")
            continue
        seen.add(key)
        if line.get("error"):
            manifest.set_status(row["id"], FAILED, error=str(line["error"])[:500])
            continue
        pcm = audio_from_line(line)
        if not pcm:
            manifest.set_status(row["id"], FAILED, error="no audio in batch result")
            continue
        _write_and_mark(manifest, out_dir, lang, voice, row, pcm)

    for key, row in rows_by_key.items():       # submitted but absent from results
        if key not in seen and row["status"] == SUBMITTED:
            manifest.set_status(row["id"], FAILED, error="missing from batch result")

    # Results are saved — delete the job as CLEANUP. The enqueued-token quota
    # counts "active" (non-terminal) jobs, so it already frees when the job
    # completes; deleting just clears the finished job + its output server-side so
    # dead jobs don't pile up until the 48h auto-expiry. Best-effort.
    if delete_after:
        try:
            batch.delete(job_name)
            log(f"deleted {job_name} (cleanup)")
        except Exception as e:  # noqa: BLE001 - cleanup is best-effort
            log(f"could not delete {job_name}: {e} (clean up later with gemini_batch_cleanup.py)")


def submit_pending(manifest: Manifest, batch: GeminiBatch, *, lang: str = "en",
                   chunk: int = 2000, log: Callable = print) -> List[str]:
    """Submit all pending words as async job(s) and return immediately.

    Job ids persist in the manifest (status -> submitted), so the machine can be
    powered off; collect later with `collect_active`. Does NOT poll.
    """
    jobs: List[str] = []
    while True:
        rows = manifest.pending(limit=chunk)
        if not rows:
            break
        jsonl = build_jsonl(rows, batch.voice, batch.instruction)
        try:
            job_name = batch.submit_file(jsonl, display_name=f"wordbank-{lang}-{rows[0]['id']}")
        except Exception as e:  # noqa: BLE001 - inspect for quota, re-raise others
            if getattr(e, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e):
                log("quota reached (429 enqueued tokens) — stopping submission. "
                    "Collect in-flight jobs to free budget, then re-run.")
                break
            raise
        # rows stay 'pending' until this commits, so a crash never half-submits
        manifest.mark_submitted([r["id"] for r in rows], job_name)
        log(f"submitted {len(rows)} words as {job_name}")
        jobs.append(job_name)
    return jobs


def collect_active(manifest: Manifest, batch: GeminiBatch, out_dir: str, *,
                   lang: str = "en", voice: Optional[str] = None,
                   poll_interval: float = 60, sleep: Callable = time.sleep,
                   log: Callable = print, delete_after: bool = False) -> dict:
    """Poll every in-flight batch (from the manifest) and write completed clips.

    Safe to run in a fresh process after a reboot — reconnects by job id.
    `delete_after` (off by default) deletes each job after collecting; it's only
    cleanup — the enqueued-token quota frees when a job completes regardless.
    Clean up jobs on request with scripts/gemini_batch_cleanup.py.
    """
    out = pathlib.Path(out_dir)
    voice = voice or batch.voice
    active = manifest.active_batch_ids()
    if not active:
        log("no in-flight batches to collect")
    for job_name in active:
        log(f"collecting batch {job_name}")
        _collect(manifest, batch, job_name, out, lang, voice, poll_interval, sleep,
                 log, delete_after)
    return manifest.counts()


def run_batch(manifest: Manifest, batch: GeminiBatch, out_dir: str, *,
              lang: str = "en", voice: Optional[str] = None, chunk: int = 2000,
              poll_interval: float = 60, sleep: Callable = time.sleep,
              log: Callable = print) -> dict:
    """Blocking convenience: submit all pending, then poll+collect in one process."""
    submit_pending(manifest, batch, lang=lang, chunk=chunk, log=log)
    return collect_active(manifest, batch, out_dir, lang=lang, voice=voice,
                          poll_interval=poll_interval, sleep=sleep, log=log)


def run_in_waves(manifest: Manifest, batch: GeminiBatch, out_dir: str, *,
                 lang: str = "en", voice: Optional[str] = None, chunk: int = 2000,
                 poll_interval: float = 60, sleep: Callable = time.sleep,
                 log: Callable = print, max_waves: int = 10_000) -> dict:
    """Quota-aware loop: collect in-flight (frees enqueued-token budget), submit
    as many words as the budget allows, repeat until nothing is pending.

    Needs the process to stay alive (collection waits for jobs to finish), so run
    it with the laptop on. Fully resumable — picks up whatever the manifest holds.
    """
    voice = voice or batch.voice
    for wave_num in range(1, max_waves + 1):
        collect_active(manifest, batch, out_dir, lang=lang, voice=voice,
                       poll_interval=poll_interval, sleep=sleep, log=log)
        if not manifest.pending(limit=1):
            log("all words processed")
            break
        jobs = submit_pending(manifest, batch, lang=lang, chunk=chunk, log=log)
        if not jobs:
            log("could not submit any job this wave — a single chunk likely exceeds "
                "the enqueued-token quota; retry with a smaller --chunk")
            break
        log(f"wave {wave_num}: {len(jobs)} job(s) in flight")
    return manifest.counts()
