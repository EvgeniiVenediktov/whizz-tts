"""Tests for the Gemini Batch path (fake client — no SDK/key/ffmpeg needed).

Audio is written as WAV from raw PCM; mapping is keyed (1:1), proven by a fake
that returns results in REVERSED order with a frame count that encodes the id.
"""
import base64
import json
import wave

from wordbank.gemini_batch import (
    VOICE_INSTRUCTION, GeminiBatch, audio_from_line, build_jsonl,
    build_request_body, collect_active, run_batch, run_in_waves, submit_pending,
    write_wav,
)
from wordbank.manifest import DONE, PENDING, Manifest, SUBMITTED


def pcm_for(key) -> bytes:
    """16-bit mono PCM whose frame count encodes the key (id) for assertions."""
    return b"\x01\x00" * (100 + int(key))


class FakeJob:
    def __init__(self, name, state, keys):
        self.name = name
        self.state = state
        self.keys = keys


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


def test_build_jsonl_keys_and_lines():
    rows = [{"id": 1, "word": "cat"}, {"id": 2, "word": "dog"}]
    lines = build_jsonl(rows, "Kore").decode().strip().splitlines()
    assert [json.loads(l)["key"] for l in lines] == ["1", "2"]


def test_audio_from_line_decodes_base64():
    b64 = base64.b64encode(b"\x01\x02\x03\x04").decode()
    line = {"key": "1", "response": {"candidates": [{"content": {"parts": [
        {"inlineData": {"data": b64}}]}}]}}
    assert audio_from_line(line) == b"\x01\x02\x03\x04"
    assert audio_from_line({"key": "1", "error": {"code": 400}}) is None


def test_write_wav_is_valid(tmp_path):
    p = tmp_path / "x.wav"
    write_wav(p, b"\x01\x00" * 240)
    assert p.read_bytes()[:4] == b"RIFF"
    with wave.open(str(p), "rb") as wf:
        assert wf.getnchannels() == 1 and wf.getframerate() == 24_000
        assert wf.getnframes() == 240


# --- end-to-end (keyed, WAV) ----------------------------------------------

def test_run_batch_writes_wavs_and_marks_done(tmp_path):
    m = Manifest(str(tmp_path / "wb.db"))
    m.add_words([f"word{i}" for i in range(5)], "en")
    fake = FakeBatch()
    counts = run_batch(m, fake, str(tmp_path / "out"),
                       lang="en", voice="Kore", sleep=lambda *_: None)
    assert counts.get(DONE) == 5
    wavs = list((tmp_path / "out").rglob("*.wav"))
    assert len(wavs) == 5
    assert all(w.read_bytes()[:4] == b"RIFF" for w in wavs)
    assert not fake.deleted           # no auto-delete by default (cleanup is on request)


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
    run_batch(m, FakeBatch(), str(tmp_path / "out"),
              lang="en", voice="Kore", sleep=lambda *_: None)
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
