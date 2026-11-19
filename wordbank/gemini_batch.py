"""Gemini Batch API path for bulk word-bank generation (async, ~half cost).

File-based JSONL with an explicit per-request KEY = manifest id, so every result
maps **1:1** to its requested word regardless of response order or omissions —
not positional. At receive time the returned PCM (24 kHz / 16-bit / mono; a WAV
payload is first unwrapped to its data chunk by gemini_tts.pcm_from_payload) is
written straight to **WAV** via the stdlib `wave` module: no ffmpeg, no trimming,
no normalization. Any OGG/Opus encoding or trimming is a separate later pass over
the WAVs on disk (see scripts/encode_wavs.py).

Text↔recording index: the manifest `word` column holds the exact requested text
and `file_path` the WAV — `SELECT word, file_path FROM words WHERE status='done'`.

google-genai is lazily imported; tests inject a fake batch client.

⚠️ Confirm with scripts/gemini_batch_smoke.py + a real key before a bulk run:
  - batch accepts the per-request TTS generationConfig on the model;
  - result audio is base64 at candidates[0].content.parts[*].inlineData.data;
  - the leading voice instruction steers delivery and is NOT spoken;
  - with --style (3.8 models), the part's speechMetadata.style is honoured. The
    docs say so (Batch API takes generateContent requests; 3.8 TTS supports
    batch and "Attach speech_metadata … to each part"), but it was not tried live.

Directions are routed per model by gemini_tts.route_direction: 3.1/2.5 get the
instruction as an inline prefix; the 3.8 models read inline text aloud, so they
get it (and/or --style) as the part's speechMetadata.style, the word verbatim.

QC (wordbank/qc.py): every collected clip is checked. A flagged one can't be
re-taken on the spot, so its row is re-queued (status RETAKE, best take so far
at qc.retake_path, outside the bank tree) for the next submission by a batch
with the same voice, model and direction — run_in_waves' next wave, or the next
batch-submit / batch-run — up to `qc_retries` times. The first passing take is
filed; if every take is flagged, the best one is filed with its flag. A re-take
counts in qc_takes, not in the row's generation attempts, and one that fails to
generate falls back to the take it was meant to replace.
"""
import base64
import json
import os
import pathlib
import tempfile
import time
import wave
from typing import Callable, Iterable, List, Optional

from . import qc
from .constants import SAMPLE_RATE
from .gemini_tts import (GeminiAudioFormatError, check_voice, new_client,
                         pcm_from_payload, route_direction)
from .manifest import DONE, FAILED, RETAKE, SUBMITTED, Manifest
from .paths import safe_filename

# NOTE: not every TTS model supports batchGenerateContent. As of this writing
# gemini-3.1-flash-tts-preview and gemini-2.5-pro-preview-tts do; the older
# gemini-2.5-flash-preview-tts does NOT. Run scripts/list_models.py to recheck.
BATCH_DEFAULT_MODEL = "gemini-3.1-flash-tts-preview"

# google.genai.types.JobState names a batch ends in: JOB_STATE_SUCCEEDED or
# JOB_STATE_PARTIALLY_SUCCEEDED (results file; per-line errors are per row), or
# JOB_STATE_FAILED / _CANCELLED / _EXPIRED (no results; jobs expire after 48h).
# Matched by suffix, so the API's raw BATCH_STATE_* spellings work too.
_TERMINAL = ("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED")

# Style direction prepended to each word. The 3.1/2.5 TTS models treat a leading
# "Say …:" clause as delivery instruction, not text to speak; the 3.8 models
# read it aloud, so the CLI never defaults to it for them (see __main__).
VOICE_INSTRUCTION = "Say the following single word clearly and naturally, in a neutral tone:"


# --- request / response helpers (pure, easily testable) --------------------

def build_request_body(word: str, voice: str, instruction: str = VOICE_INSTRUCTION,
                       style: Optional[str] = None) -> dict:
    """Raw REST GenerateContentRequest body (camelCase) for one word's audio.
    `instruction` is an inline prefix; `style` goes in the part's
    speechMetadata (the word stays verbatim). Which one a model gets is
    GeminiBatch's call (route_direction), not this builder's."""
    part = {"text": f"{instruction} {word}" if instruction else word}
    if style:
        part["speechMetadata"] = {"style": style}
    return {
        "contents": [{"parts": [part], "role": "user"}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}
            },
        },
    }


def build_jsonl(rows, voice: str, instruction: str = VOICE_INSTRUCTION,
                style: Optional[str] = None) -> bytes:
    """One JSONL line per word: {"key": "<id>", "request": {...}}."""
    lines = [
        json.dumps({"key": str(r["id"]),
                    "request": build_request_body(r["word"], voice, instruction, style)})
        for r in rows
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def audio_from_line(line: dict) -> Optional[bytes]:
    """Decode one output JSONL line's base64 audio to raw PCM; None if absent.
    Raises GeminiAudioFormatError for audio that isn't 24 kHz/16-bit/mono."""
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
            return pcm_from_payload(base64.b64decode(blob["data"]),
                                    blob.get("mimeType") or blob.get("mime_type"))
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
                 instruction: str = VOICE_INSTRUCTION, style: Optional[str] = None,
                 client=None):
        """`instruction`/`style` are routed for `model` (route_direction):
        afterwards `instruction` is what is prepended inline and `style` what
        goes in speechMetadata — on a 3.8 model the instruction moves into the
        style, so it is never spoken. A style for a model that takes none
        raises GeminiStyleUnsupportedError here, before anything is submitted."""
        self.model = model
        self.voice = voice
        self.style, self.instruction = route_direction(model, instruction, style)
        self._client = client

    @property
    def direction(self) -> Optional[str]:
        """The delivery direction this batch sends (style or inline), recorded
        as each row's `style` provenance."""
        return self.style or self.instruction or None

    def _get_client(self):
        if self._client is None:
            self._client = new_client()
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
        # a locale voice on a model not verified with it fails before the
        # upload (not in __init__: collect-only batches carry no real model)
        check_voice(self.voice, self.model)
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

    @staticmethod
    def _state_name(job) -> str:
        state = getattr(job, "state", "")
        return getattr(state, "name", None) or str(state)

    def is_done(self, job) -> bool:
        return self._state_name(job).endswith(_TERMINAL)

    def no_results_reason(self, job) -> Optional[str]:
        """Why a finished job has no results to collect (naming its state), or
        None when it has. Only (PARTIALLY_)SUCCEEDED jobs carry a results file."""
        state = self._state_name(job)
        if not state.endswith("SUCCEEDED"):
            msg = getattr(getattr(job, "error", None), "message", None)
            return f"batch job {state}" + (f": {msg}" if msg else "")
        if not getattr(getattr(job, "dest", None), "file_name", None):
            return f"batch job {state} has no results file"
        return None

    def result_lines(self, job) -> Iterable[dict]:
        content = self._get_client().files.download(file=job.dest.file_name)
        for line in content.decode("utf-8").splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)

    def delete(self, job_name: str) -> None:
        self._get_client().batches.delete(name=job_name)


# --- pipeline --------------------------------------------------------------

def clip_rel(lang: str, voice: str, row) -> str:
    """A word clip's path RELATIVE to the bank root (out_dir), so the manifest
    is location-independent: the server resolves it under WAV_ROOT/OGG_ROOT,
    which differ between dev (output/words) and the container (/data/words)."""
    return os.path.join(lang, f"gemini-{voice}", str(row["id"] // 1000),
                        f"{row['id']}_{safe_filename(row['word'])}.wav")


def _write_and_mark(manifest: Manifest, out_dir: pathlib.Path, lang: str,
                    voice: str, row, pcm: bytes, *, model: Optional[str] = None,
                    style: Optional[str] = None, check: Optional[qc.Check] = None,
                    qc_takes: int = 0) -> None:
    """File a take as the row's clip, with its QC result (`check`, computed
    here when not given) and the number of re-takes it took."""
    rel = clip_rel(lang, voice, row)
    path = out_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    write_wav_from_pcm_bytes(path, pcm)
    # model/style: the sync runner passes them; batch collect leaves them as
    # submit_pending recorded them (see Manifest.set_status).
    manifest.set_status(row["id"], DONE, file_path=rel,
                        billed_chars=len(row["word"]), voice=voice,
                        model=model, style=style, check=check or qc.check_pcm(pcm),
                        qc_takes=qc_takes)


def _settle_take(manifest: Manifest, out_dir: pathlib.Path, row, pcm: Optional[bytes],
                 *, voice: str, qc_retries: int) -> Optional[str]:
    """QC a collected take (`pcm`; None when the request produced none) against
    the row's best take so far, if it is a re-take, and file the outcome:
    "passed" / "resolved" (passed on a re-take) / "flagged" (re-takes used up,
    the best take filed with its flag) / "requeued" (status RETAKE). None when
    there is no take at all: the caller fails the row as before QC."""
    lang, voice = row["lang"], row["voice"] or voice
    held = qc.retake_path(manifest.path, row["id"])
    takes = []
    if row["qc_takes"] and held.is_file():
        takes.append((qc.read_pcm(held), qc.check_wav(held)))
    if pcm:
        takes.append((pcm, qc.check_pcm(pcm)))
    if not takes:
        return None
    best, check = qc.best_take(*takes)
    if check.flagged and row["qc_takes"] < qc_retries:
        held.parent.mkdir(parents=True, exist_ok=True)
        write_wav_from_pcm_bytes(held, best)
        manifest.set_status(row["id"], RETAKE, file_path=clip_rel(lang, voice, row),
                            billed_chars=len(row["word"]), voice=voice, check=check,
                            qc_takes=row["qc_takes"] + 1)
        return "requeued"
    _write_and_mark(manifest, out_dir, lang, voice, row, best, check=check,
                    qc_takes=row["qc_takes"])
    held.unlink(missing_ok=True)
    if check.flagged:
        return "flagged"
    return "resolved" if row["qc_takes"] else "passed"


def _collect(manifest: Manifest, batch: GeminiBatch, job_name: str,
             out_dir: pathlib.Path, lang: str, voice: str,
             poll_interval: float, sleep: Callable, log: Callable,
             delete_after: bool = False, qc_retries: int = qc.DEFAULT_RETRIES) -> None:
    while True:
        job = batch.poll(job_name)
        if batch.is_done(job):
            break
        log(f"{job_name}: {getattr(job, 'state', '?')} — waiting {poll_interval}s")
        sleep(poll_interval)

    # Rows already settled (a collect interrupted half-way) are left alone.
    rows_by_key = {str(r["id"]): r for r in manifest.get_by_batch(job_name)
                   if r["status"] == SUBMITTED}
    outcomes = {}

    def settle(row, pcm=None, error=None):
        # A re-take that produced nothing falls back to the take it was to replace.
        outcome = _settle_take(manifest, out_dir, row, pcm, voice=voice, qc_retries=qc_retries)
        if outcome is None:
            manifest.set_status(row["id"], FAILED, error=error[:500])
            outcome = "failed"
        outcomes[outcome] = outcomes.get(outcome, 0) + 1

    why = batch.no_results_reason(job)
    if why:
        # A FAILED/CANCELLED/EXPIRED job has nothing to download. Fail its words
        # (the submission already spent one attempt) so pending() retries them
        # within max_attempts, rather than leaving them SUBMITTED for every
        # later collect to trip over again. A QC re-take keeps its held take.
        for row in rows_by_key.values():
            settle(row, error=why)
        log(f"{job_name}: {why} — {_tally(outcomes)}")
        return

    seen = set()
    for line in batch.result_lines(job):
        key = str(line.get("key"))
        row = rows_by_key.get(key)
        if row is None:
            log(f"{job_name}: result for unknown or settled key {key!r} — skipped")
            continue
        seen.add(key)
        if line.get("error"):
            settle(row, error=str(line["error"]))
            continue
        try:
            pcm = audio_from_line(line)
        except GeminiAudioFormatError as e:
            settle(row, error=str(e))
            continue
        settle(row, pcm, error="no audio in batch result")

    for key, row in rows_by_key.items():       # submitted but absent from results
        if key not in seen:
            settle(row, error="missing from batch result")
    log(f"{job_name}: {_tally(outcomes)}")

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


def _pending(manifest: Manifest, batch: GeminiBatch, lang: str, limit: int):
    """This batch's work: `lang` rows in its voice (or unattributed) still to
    generate, and RETAKE rows recorded with its model and direction."""
    return manifest.pending(limit=limit, lang=lang, voice=batch.voice,
                            retake=(batch.model, batch.direction))


def _tally(outcomes: dict) -> str:
    text = ", ".join(f"{n} {k}" for k, n in sorted(outcomes.items())) or "nothing to settle"
    if outcomes.get("requeued"):
        text += (" (requeued = flagged by QC: status retake until the next submission "
                 "with the same voice, model and direction re-takes it)")
    return text


def submit_pending(manifest: Manifest, batch: GeminiBatch, *, lang: str = "en",
                   chunk: int = 2000, log: Callable = print) -> List[str]:
    """Submit all pending `lang` words in the batch's voice as async job(s),
    QC re-takes queued for its model and direction included, and return
    immediately.

    Job ids persist in the manifest (status -> submitted), so the machine can be
    powered off; collect later with `collect_active`. Does NOT poll.
    """
    jobs: List[str] = []
    while True:
        rows = _pending(manifest, batch, lang, chunk)
        if not rows:
            break
        jsonl = build_jsonl(rows, batch.voice, batch.instruction, batch.style)
        try:
            job_name = batch.submit_file(jsonl, display_name=f"wordbank-{lang}-{rows[0]['id']}")
        except Exception as e:  # noqa: BLE001 - inspect for quota, re-raise others
            if getattr(e, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e):
                log("quota reached (429 enqueued tokens) — stopping submission. "
                    "Collect in-flight jobs to free budget, then re-run.")
                break
            raise
        # rows stay 'pending' until this commits, so a crash never half-submits
        manifest.mark_submitted([r["id"] for r in rows], job_name,
                                model=batch.model, style=batch.direction)
        log(f"submitted {len(rows)} words as {job_name}")
        jobs.append(job_name)
    return jobs


def collect_active(manifest: Manifest, batch: GeminiBatch, out_dir: str, *,
                   lang: str = "en", voice: Optional[str] = None,
                   poll_interval: float = 60, sleep: Callable = time.sleep,
                   log: Callable = print, delete_after: bool = False,
                   qc_retries: int = qc.DEFAULT_RETRIES) -> dict:
    """Poll every in-flight batch (from the manifest) and write completed clips.

    Safe to run in a fresh process after a reboot — reconnects by job id.
    Clips are filed under each row's own language and voice (`voice` only for
    rows from before the voice key). A clip flagged by QC is re-queued (status
    RETAKE) while it has re-takes left of `qc_retries`.
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
                 log, delete_after, qc_retries)
    return manifest.counts()


def run_batch(manifest: Manifest, batch: GeminiBatch, out_dir: str, *,
              lang: str = "en", voice: Optional[str] = None, chunk: int = 2000,
              poll_interval: float = 60, sleep: Callable = time.sleep,
              log: Callable = print, qc_retries: int = qc.DEFAULT_RETRIES) -> dict:
    """Blocking convenience: submit all pending, then poll+collect in one process.
    QC re-takes it queues wait for the next submission (run_in_waves loops)."""
    submit_pending(manifest, batch, lang=lang, chunk=chunk, log=log)
    return collect_active(manifest, batch, out_dir, lang=lang, voice=voice,
                          poll_interval=poll_interval, sleep=sleep, log=log,
                          qc_retries=qc_retries)


def run_in_waves(manifest: Manifest, batch: GeminiBatch, out_dir: str, *,
                 lang: str = "en", voice: Optional[str] = None, chunk: int = 2000,
                 poll_interval: float = 60, sleep: Callable = time.sleep,
                 log: Callable = print, max_waves: int = 10_000,
                 qc_retries: int = qc.DEFAULT_RETRIES) -> dict:
    """Quota-aware loop: collect in-flight (frees enqueued-token budget), submit
    as many words as the budget allows, repeat until nothing is pending — QC
    re-takes included, so it ends with every clip passed or out of re-takes.

    Needs the process to stay alive (collection waits for jobs to finish), so run
    it with the laptop on. Fully resumable — picks up whatever the manifest holds.
    """
    voice = voice or batch.voice
    for wave_num in range(1, max_waves + 1):
        collect_active(manifest, batch, out_dir, lang=lang, voice=voice,
                       poll_interval=poll_interval, sleep=sleep, log=log,
                       qc_retries=qc_retries)
        if not _pending(manifest, batch, lang, 1):
            log("all words processed")
            break
        jobs = submit_pending(manifest, batch, lang=lang, chunk=chunk, log=log)
        if not jobs:
            log("could not submit any job this wave — a single chunk likely exceeds "
                "the enqueued-token quota; retry with a smaller --chunk")
            break
        log(f"wave {wave_num}: {len(jobs)} job(s) in flight")
    return manifest.counts()
