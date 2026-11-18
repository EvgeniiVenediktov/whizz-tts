"""Tests for the Gemini Batch path (fake client — no SDK/key/ffmpeg needed).

Audio is written as WAV from raw PCM; mapping is keyed (1:1), proven by a fake
that returns results in REVERSED order with a frame count that encodes the id.
"""
import base64
import json
import wave
from types import SimpleNamespace

import pytest

from tests.test_gemini_payload import gemini_lite_wav
from wordbank.gemini_batch import (
    VOICE_INSTRUCTION, GeminiBatch, audio_from_line, build_request_body, collect_active, run_batch, run_in_waves, submit_pending,
    write_wav_from_pcm_bytes,
)
from wordbank.manifest import DONE, FAILED, PENDING, Manifest, SUBMITTED


def quiet(*_):
    pass


def pcm_for(key) -> bytes:
    """16-bit mono PCM whose frame count encodes the key (id) for assertions."""
    return b"\x01\x00" * (100 + int(key))


class FakeJob:
    def __init__(self, name, state, keys):
        self.name = name
        self.state = state
        self.keys = keys
        self.dest = SimpleNamespace(file_name=f"files/{name}-results")


class FakeBatch(GeminiBatch):
    """In-memory batch; jobs 'succeed' immediately and return results REVERSED."""
    def __init__(self, voice="Kore"):
        super().__init__(voice=voice)
        self._jobs = {}
        self._n = 0
        self.deleted = []

    def submit_file(self, jsonl, display_name):
        self._n += 1
        name = f"job-{self._n}"
        self._jobs[name] = [json.loads(l)["key"]
                            for l in jsonl.decode().splitlines() if l.strip()]
        return name

    def poll(self, job_name):
        return FakeJob(job_name, "JOB_STATE_SUCCEEDED", self._jobs[job_name])

    def is_done(self, job):
        return True

    def result_lines(self, job):
        for key in reversed(job.keys):          # reversed → proves keyed mapping
            b64 = base64.b64encode(pcm_for(key)).decode()
            yield {"key": key,
                   "response": {"candidates": [{"content": {"parts": [
                       {"inlineData": {"data": b64, "mimeType": "audio/L16"}}]}}]}}

    def delete(self, job_name):
        self.deleted.append(job_name)


# --- pure helpers ----------------------------------------------------------

def test_build_request_body_shape():
    r = build_request_body("cat", "Kore")
    text = r["contents"][0]["parts"][0]["text"]
    assert text == f"{VOICE_INSTRUCTION} cat"
    gc = r["generationConfig"]
    assert gc["responseModalities"] == ["AUDIO"]
    assert gc["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Kore"


def test_build_request_body_bare_word():
    r = build_request_body("cat", "Kore", instruction="")
    assert r["contents"][0]["parts"][0]["text"] == "cat"


def test_audio_from_line_decodes_base64():
    b64 = base64.b64encode(b"\x01\x02\x03\x04").decode()
    line = {"key": "1", "response": {"candidates": [{"content": {"parts": [
        {"inlineData": {"data": b64}}]}}]}}
    assert audio_from_line(line) == b"\x01\x02\x03\x04"
    assert audio_from_line({"key": "1", "error": {"code": 400}}) is None


def test_audio_from_line_unwraps_a_wav_payload():
    """gemini-3.8-flash-lite-tts style: a whole WAV (+ C2PA chunk) as the data."""
    b64 = base64.b64encode(gemini_lite_wav(b"\x01\x02\x03\x04")).decode()
    line = {"key": "1", "response": {"candidates": [{"content": {"parts": [
        {"inlineData": {"data": b64, "mimeType": "audio/wav"}}]}}]}}
    assert audio_from_line(line) == b"\x01\x02\x03\x04"


def test_write_wav_is_valid(tmp_path):
    p = tmp_path / "x.wav"
    write_wav_from_pcm_bytes(p, b"\x01\x00" * 240)
    assert p.read_bytes()[:4] == b"RIFF"
    with wave.open(str(p), "rb") as wf:
        assert wf.getnchannels() == 1 and wf.getframerate() == 24_000
        assert wf.getnframes() == 240


# --- end-to-end (keyed, WAV) ----------------------------------------------

def test_collect_deletes_only_when_opted_in(tmp_path):
    m = Manifest(str(tmp_path / "wb.db"))
    m.add_words(["alpha", "beta"], "en")
    fake = FakeBatch()
    submit_pending(m, fake, lang="en", chunk=10)
    collect_active(m, fake, str(tmp_path / "out"), lang="en", voice="Kore",
                   sleep=lambda *_: None, delete_after=True)
    assert fake.deleted               # opt-in deletion still works


def test_keyed_one_to_one_correspondence(tmp_path):
    """Each word's file must hold ITS audio even though results came reversed."""
    m = Manifest(str(tmp_path / "wb.db"))
    m.add_words(["alpha", "beta", "gamma", "delta"], "en")
    fake = FakeBatch()
    counts = run_batch(m, fake, str(tmp_path / "out"),
                       lang="en", voice="Kore", sleep=lambda *_: None)
    assert counts.get(DONE) == 4
    assert not fake.deleted           # no auto-delete by default (cleanup is on request)
    out = tmp_path / "out"
    for r in m.iter_rows():
        assert r["status"] == DONE
        assert f"{r['id']}_" in r["file_path"]          # filename carries the id
        assert not r["file_path"].startswith("/")        # stored relative to bank root
        with wave.open(str(out / r["file_path"]), "rb") as wf:
            # frame count encodes the id → confirms the right audio landed here
            assert wf.getnframes() == 100 + r["id"]
    m.close()


def test_submit_then_collect_separately(tmp_path):
    """Submit and exit (power-off), then collect later — the overnight flow."""
    m = Manifest(str(tmp_path / "wb.db"))
    m.add_words(["one", "two", "three"], "en")
    fake = FakeBatch()
    jobs = submit_pending(m, fake, lang="en", chunk=2)
    assert len(jobs) == 2                                 # 3 words, chunk 2 → 2 jobs
    assert m.counts().get(SUBMITTED) == 3
    assert not list((tmp_path / "out").rglob("*.wav"))    # nothing written at submit

    counts = collect_active(m, fake, str(tmp_path / "out"),
                            lang="en", voice="Kore", sleep=lambda *_: None)
    assert counts.get(DONE) == 3
    assert len(list((tmp_path / "out").rglob("*.wav"))) == 3
    m.close()


# --- quota (429) handling ---------------------------------------------------

class QuotaFakeBatch(FakeBatch):
    """Models the enqueued-token budget: at most `max_inflight` jobs counted at
    once. submit_file 429s when full; the budget frees when a job COMPLETES (it
    counts only active/non-terminal jobs) — so the wave loop drains because it
    waits for completion before submitting the next batch. (delete is cleanup.)
    """
    def __init__(self, max_inflight=2, voice="Kore"):
        super().__init__(voice=voice)
        self.max_inflight = max_inflight
        self.inflight = 0

    def submit_file(self, jsonl, display_name):
        if self.inflight >= self.max_inflight:
            e = Exception("429 RESOURCE_EXHAUSTED: enqueued tokens")
            e.code = 429
            raise e
        self.inflight += 1
        return super().submit_file(jsonl, display_name)

    def result_lines(self, job):
        yield from super().result_lines(job)
        self.inflight -= 1            # job completed & collected → no longer active


def test_submit_pending_stops_gracefully_on_429(tmp_path):
    m = Manifest(str(tmp_path / "wb.db"))
    m.add_words([f"w{i}" for i in range(6)], "en")     # 6 words, chunk 2 → 3 chunks
    jobs = submit_pending(m, QuotaFakeBatch(max_inflight=2), lang="en", chunk=2)
    assert len(jobs) == 2                               # 3rd chunk 429'd → stop, no crash
    assert m.counts().get(SUBMITTED) == 4
    assert m.counts().get(PENDING) == 2                 # un-submitted chunk stays pending
    m.close()


def test_run_in_waves_drains_everything_under_quota(tmp_path):
    m = Manifest(str(tmp_path / "wb.db"))
    m.add_words([f"w{i}" for i in range(10)], "en")     # 10 words, chunk 2, budget 2 jobs
    counts = run_in_waves(m, QuotaFakeBatch(max_inflight=2), str(tmp_path / "out"),
                          lang="en", voice="Kore", chunk=2, sleep=lambda *_: None)
    assert counts.get(DONE) == 10                       # all processed despite the cap
    assert len(list((tmp_path / "out").rglob("*.wav"))) == 10
    m.close()


# --- WAV payloads (gemini-3.8-flash-lite-tts) --------------------------------

class WavPayloadBatch(FakeBatch):
    """Results carry whole WAV files (+ C2PA chunk); key `bad` is stereo."""
    bad = None

    def result_lines(self, job):
        for key in job.keys:
            wav = gemini_lite_wav(pcm_for(key), channels=2 if key == self.bad else 1)
            yield {"key": key, "response": {"candidates": [{"content": {"parts": [
                {"inlineData": {"data": base64.b64encode(wav).decode(),
                                "mimeType": "audio/wav"}}]}}]}}


def test_collect_writes_only_the_data_chunk_and_fails_unconvertible_audio(tmp_path):
    m = Manifest(str(tmp_path / "wb.db"))
    m.add_words(["alpha", "beta", "gamma"], "en")
    fake = WavPayloadBatch()
    fake.bad = str(m.pending()[1]["id"])
    run_batch(m, fake, str(tmp_path / "out"), lang="en", voice="Kore",
              sleep=lambda *_: None, log=quiet)
    rows = {r["word"]: r for r in m.iter_rows()}
    assert rows["beta"]["status"] == FAILED and "2 ch" in rows["beta"]["error"]
    for word in ("alpha", "gamma"):
        r = rows[word]
        assert r["status"] == DONE
        with wave.open(str(tmp_path / "out" / r["file_path"]), "rb") as wf:
            assert wf.getnframes() == 100 + r["id"]     # no header / C2PA frames
    m.close()


# --- jobs that end without results (FAILED / CANCELLED / EXPIRED) -------------

class FakeGenaiClient:
    """The google-genai surface the REAL GeminiBatch touches. Each created job
    ends in the next of `states` (the last repeats) as a real types.BatchJob; a
    SUCCEEDED one gets a results file answering every key with pcm_for(key)."""

    def __init__(self, *states):
        self.states = states
        self.jobs, self.downloads, self.uploads = {}, {}, {}
        self.files = SimpleNamespace(upload=self._upload,
                                     download=lambda file: self.downloads[file])
        self.batches = SimpleNamespace(create=self._create, get=lambda name: self.jobs[name])

    def _upload(self, file, config):
        name = f"files/in-{len(self.uploads)}"
        with open(file, encoding="utf-8") as f:
            self.uploads[name] = [json.loads(l)["key"] for l in f if l.strip()]
        return SimpleNamespace(name=name, state="ACTIVE")

    def _create(self, model, src, config):
        from google.genai import types
        state = self.states[min(len(self.jobs), len(self.states) - 1)]
        name = f"batches/{len(self.jobs)}"
        job = types.BatchJob(name=name, state=state)
        if state == types.JobState.JOB_STATE_SUCCEEDED:
            job.dest = types.BatchJobDestination(file_name=f"files/out-{len(self.jobs)}")
            self.downloads[job.dest.file_name] = "".join(json.dumps({
                "key": k, "response": {"candidates": [{"content": {"parts": [{"inlineData": {
                    "data": base64.b64encode(pcm_for(k)).decode(),
                    "mimeType": "audio/L16;codec=pcm;rate=24000"}}]}}]}}) + "\n"
                for k in self.uploads[src]).encode()
        elif state == types.JobState.JOB_STATE_FAILED:
            job.error = types.JobError(code=13, message="internal error")
        self.jobs[name] = job
        return job


def test_collect_fails_words_of_jobs_without_results_and_keeps_going(tmp_path):
    """An EXPIRED/FAILED job has dest=None: collecting it used to raise
    AttributeError and leave its words SUBMITTED, so every later collect
    crashed on the same job and the good job's audio was never written."""
    js = pytest.importorskip("google.genai.types").JobState
    m = Manifest(str(tmp_path / "wb.db"))
    m.add_words(list("abcdef"), "en")
    batch = GeminiBatch(client=FakeGenaiClient(
        js.JOB_STATE_EXPIRED, js.JOB_STATE_FAILED, js.JOB_STATE_SUCCEEDED))
    submit_pending(m, batch, lang="en", chunk=2, log=quiet)       # 3 jobs of 2
    counts = collect_active(m, batch, str(tmp_path / "out"), lang="en",
                            sleep=lambda *_: None, log=quiet)

    rows = {r["word"]: r for r in m.iter_rows()}
    assert [rows[w]["status"] for w in "abcdef"] == [FAILED] * 4 + [DONE] * 2
    assert rows["a"]["error"] == rows["b"]["error"] == "batch job JOB_STATE_EXPIRED"
    assert rows["c"]["error"] == "batch job JOB_STATE_FAILED: internal error"
    assert len(list((tmp_path / "out").rglob("*.wav"))) == 2
    assert m.active_batch_ids() == []
    assert [r["word"] for r in m.pending()] == list("abcd")       # normal retry path
    assert collect_active(m, batch, str(tmp_path / "out"), lang="en",
                          sleep=lambda *_: None, log=quiet) == counts
    m.close()


def test_jobs_that_always_expire_stop_at_the_attempt_budget(tmp_path):
    """Each submission is one attempt, so words whose jobs keep expiring are
    retried up to max_attempts (3) and then left FAILED — no endless waves."""
    js = pytest.importorskip("google.genai.types").JobState
    m = Manifest(str(tmp_path / "wb.db"))
    m.add_words(["a", "b"], "en")
    client = FakeGenaiClient(js.JOB_STATE_EXPIRED)
    counts = run_in_waves(m, GeminiBatch(client=client), str(tmp_path / "out"),
                          lang="en", chunk=2, sleep=lambda *_: None, log=quiet)
    assert counts == {FAILED: 2}
    assert len(client.jobs) == 3
    assert [r["attempts"] for r in m.iter_rows()] == [3, 3]
    m.close()


@pytest.mark.parametrize("state, dest, reason", [
    ("JOB_STATE_SUCCEEDED", "files/out", None),
    ("JOB_STATE_PARTIALLY_SUCCEEDED", "files/out", None),
    ("JOB_STATE_SUCCEEDED", None, "batch job JOB_STATE_SUCCEEDED has no results file"),
    ("JOB_STATE_CANCELLED", None, "batch job JOB_STATE_CANCELLED"),
    ("JOB_STATE_EXPIRED", None, "batch job JOB_STATE_EXPIRED"),
])
def test_no_results_reason(state, dest, reason):
    types = pytest.importorskip("google.genai.types")
    job = types.BatchJob(state=state, dest=dest and types.BatchJobDestination(file_name=dest))
    batch = GeminiBatch(client=object())
    assert batch.is_done(job)
    assert batch.no_results_reason(job) == reason
