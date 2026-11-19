"""Tests for the Gemini Batch path (fake client — no SDK/key/ffmpeg needed).

Audio is written as WAV from raw PCM; mapping is keyed (1:1), proven by a fake
that returns results in REVERSED order with a frame count that encodes the id.
"""
import base64
import json
import wave
from types import SimpleNamespace

import pytest

from tests.conftest import clip_pcm16
from tests.test_gemini_payload import gemini_lite_wav
from wordbank import qc
from wordbank.gemini_batch import (
    VOICE_INSTRUCTION, GeminiBatch, audio_from_line, build_request_body, collect_active, run_batch, run_in_waves, submit_pending,
    write_wav_from_pcm_bytes,
)
from wordbank.manifest import DONE, FAILED, PENDING, RETAKE, Manifest, SUBMITTED


def quiet(*_):
    pass


def pcm_for(key) -> bytes:
    """16-bit mono PCM whose frame count encodes the key (id) for assertions:
    ~0.5 s, ending quiet, so it passes QC."""
    return b"\x01\x00" * (12_000 + int(key))


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
            assert wf.getnframes() == 12_000 + r["id"]
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
            assert wf.getnframes() == 12_000 + r["id"]     # no header / C2PA frames
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


# --- style directions (3.8 models): speechMetadata, never an inline prefix ----

STYLE = "read the word exactly as written, plain and neutral"


def test_build_request_body_with_style_keeps_the_word_verbatim():
    r = build_request_body("laughing", "Kore", instruction="", style=STYLE)
    assert r["contents"][0]["parts"] == [{"text": "laughing", "speechMetadata": {"style": STYLE}}]
    assert r["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"] == \
        {"voiceName": "Kore"}


def test_gemini_batch_never_sends_an_inline_instruction_to_a_38_model():
    """Even the library default VOICE_INSTRUCTION moves into the style there;
    3.1 keeps it inline, as before."""
    batch = GeminiBatch(model="gemini-3.8-flash-tts")
    assert (batch.style, batch.instruction) == (VOICE_INSTRUCTION, "")
    batch = GeminiBatch(model="gemini-3.1-flash-tts-preview")
    assert (batch.style, batch.instruction) == (None, VOICE_INSTRUCTION)


class RecordingBatch(FakeBatch):
    """FakeBatch that keeps every submitted JSONL line."""
    def __init__(self, **kw):
        GeminiBatch.__init__(self, **kw)
        self._jobs, self._n, self.deleted, self.lines = {}, 0, [], []

    def submit_file(self, jsonl, display_name):
        self.lines += [json.loads(l) for l in jsonl.decode().splitlines() if l.strip()]
        return super().submit_file(jsonl, display_name)


def test_style_batch_sends_speech_metadata_and_records_provenance(tmp_path):
    m = Manifest(str(tmp_path / "wb.db"))
    m.add_words(["laughing", "haha"], "en-gb")
    batch = RecordingBatch(model="gemini-3.8-flash-lite-tts", voice="en-gb-tutor-11",
                           instruction="", style=STYLE)
    submit_pending(m, batch, lang="en-gb", chunk=10, log=quiet)
    assert [l["request"]["contents"][0]["parts"] for l in batch.lines] == [
        [{"text": w, "speechMetadata": {"style": STYLE}}] for w in ("laughing", "haha")]

    for r in m.iter_rows():                 # recorded at submit time...
        assert (r["status"], r["model"], r["style"]) == (SUBMITTED, "gemini-3.8-flash-lite-tts", STYLE)

    # ...so a collect in a later process, by a batch that knows nothing of the
    # style (batch-collect takes no --style), keeps it
    collector = RecordingBatch(model="gemini-3.8-flash-lite-tts", instruction="")
    collector._jobs = batch._jobs
    collect_active(m, collector, str(tmp_path / "out"), lang="en-gb",
                   voice="en-gb-tutor-11", sleep=lambda *_: None, log=quiet)
    for r in m.iter_rows():
        assert (r["status"], r["model"], r["style"]) == (DONE, "gemini-3.8-flash-lite-tts", STYLE)
        assert r["file_path"].startswith("en-gb/gemini-en-gb-tutor-11/0/")
    m.close()


def test_cli_direction_args():
    """(instruction, style) from --instruction/--style; an unset --instruction
    defaults to VOICE_INSTRUCTION only where it stays unspoken."""
    from wordbank.__main__ import _direction_args

    def direction(model, instruction=None, style=None):
        return _direction_args(SimpleNamespace(model=model, instruction=instruction, style=style))

    assert direction("gemini-3.8-flash-tts", style=STYLE) == ("", STYLE)
    assert direction("gemini-3.8-flash-tts", instruction="") == ("", None)
    assert direction("gemini-3.1-flash-tts-preview") == (VOICE_INSTRUCTION, None)
    with pytest.raises(SystemExit, match="--style needs a style-capable"):
        direction("gemini-3.1-flash-tts-preview", style=STYLE)


def test_batch_run_on_a_38_model_needs_an_explicit_direction(tmp_path):
    """batch-run used to send VOICE_INSTRUCTION inline to whatever model it was
    given; on 3.8 that instruction was read out in every clip."""
    from wordbank.__main__ import main
    words = tmp_path / "w.txt"
    words.write_text("laughing\n")
    db = tmp_path / "wb.db"
    with pytest.raises(SystemExit, match="reads inline instructions aloud"):
        main(["--db", str(db), "--out", str(tmp_path / "out"), "batch-run",
              "--words", str(words), "--model", "gemini-3.8-flash-tts"])
    assert not db.exists()                                  # refused before ingesting


# --- voices: one row per (word, voice) -----------------------------------------

def test_a_second_voice_is_submitted_without_touching_the_first(tmp_path):
    m = Manifest(str(tmp_path / "wb.db"))
    m.add_words(["cat", "dog"], "en", voice="Kore")
    run_batch(m, FakeBatch(voice="Kore"), str(tmp_path / "out"), lang="en",
              sleep=lambda *_: None, log=quiet)
    assert m.add_words(["cat", "dog", "emu"], "en", voice="Puck") == 3

    puck = RecordingBatch(voice="Puck")
    run_batch(m, puck, str(tmp_path / "out"), lang="en", sleep=lambda *_: None, log=quiet)
    assert [l["request"]["contents"][0]["parts"][0]["text"].split()[-1] for l in puck.lines] \
        == ["cat", "dog", "emu"]
    rows = {(r["word"], r["voice"]): r for r in m.iter_rows()}
    assert len(rows) == 5 and {r["status"] for r in rows.values()} == {DONE}
    assert rows[("cat", "Kore")]["file_path"].startswith("en/gemini-Kore/")
    assert rows[("cat", "Puck")]["file_path"].startswith("en/gemini-Puck/")
    m.close()


# --- QC: flagged takes are re-queued for the next submission ------------------------

class ScriptedBatch(FakeBatch):
    """Answers a word's submissions from `takes[word]` in turn (PCM, or a dict
    for an error line); other words, and words whose list ran out, get a
    passing clip. `sent` lists the words of each submitted job."""
    def __init__(self, takes, **kw):
        super().__init__(**kw)
        self.takes, self.sent, self._words = takes, [], {}

    def submit_file(self, jsonl, display_name):
        words = [json.loads(l)["request"]["contents"][0]["parts"][0]["text"].split()[-1]
                 for l in jsonl.decode().splitlines() if l.strip()]
        name = super().submit_file(jsonl, display_name)
        self.sent.append(words)
        self._words[name] = words
        return name

    def result_lines(self, job):
        for key, word in zip(job.keys, self._words[job.name]):
            take = (self.takes.get(word) or [clip_pcm16(seed=int(key))]).pop(0)
            if isinstance(take, dict):
                yield {"key": key, **take}
                continue
            yield {"key": key, "response": {"candidates": [{"content": {"parts": [{"inlineData": {
                "data": base64.b64encode(take).decode(), "mimeType": "audio/L16"}}]}}]}}


def test_waves_retake_flagged_clips_without_spending_generation_attempts(tmp_path):
    """haha passes on its first re-take; russian's three takes are all flagged,
    so the best (plausible length, quietest end) is filed with its flag."""
    m = Manifest(str(tmp_path / "wb.db"))
    m.add_words(["cat", "haha", "russian"], "en", voice="Kore")
    best = clip_pcm16(end_db=-30)
    batch = ScriptedBatch({"haha": [clip_pcm16(end_db=-25)],
                           "russian": [clip_pcm16(end_db=-20), best, clip_pcm16(3.5)]})
    run_in_waves(m, batch, str(tmp_path / "out"), lang="en", chunk=10,
                 sleep=lambda *_: None, log=quiet)

    assert batch.sent == [["cat", "haha", "russian"], ["haha", "russian"], ["russian"]]
    rows = {r["word"]: r for r in m.iter_rows()}
    assert [(rows[w]["status"], rows[w]["review"], rows[w]["qc_takes"], rows[w]["attempts"])
            for w in ("cat", "haha", "russian")] == [
        (DONE, None, 0, 1), (DONE, None, 1, 1), (DONE, "ends on sound (-30 dB)", 2, 1)]
    assert qc.read_pcm(tmp_path / "out" / rows["russian"]["file_path"]) == best
    assert not list((tmp_path / "takes").rglob("*.wav"))        # held takes cleaned up
    m.close()


def test_collect_alone_leaves_a_flagged_row_queued_for_its_own_batch(tmp_path):
    """batch-collect without waves: the row is flagged and RETAKE (not served,
    its take held outside the bank tree), and says so; only a batch with the
    take's voice, model and direction re-takes it, and a re-take that fails
    falls back to the held take instead of failing the row."""
    m = Manifest(str(tmp_path / "wb.db"))
    m.add_words(["haha"], "en", voice="Kore")
    logs = []
    batch = ScriptedBatch({"haha": [clip_pcm16(end_db=-25), {"error": {"code": 500}}]})
    run_batch(m, batch, str(tmp_path / "out"), lang="en", sleep=lambda *_: None,
              log=logs.append)
    [row] = m.iter_rows()
    assert (row["status"], row["review"], row["qc_takes"], row["attempts"]) == \
        (RETAKE, "ends on sound (-25 dB)", 1, 1)
    assert any("1 requeued" in line and "same voice, model and direction" in line
               for line in logs)
    assert not list((tmp_path / "out").rglob("*.wav"))
    assert qc.retake_path(m.path, row["id"]).is_file()

    other_direction = ScriptedBatch({})
    other_direction.instruction = "Say it slowly:"
    for other in (ScriptedBatch({}, voice="Puck"), other_direction):
        assert submit_pending(m, other, lang="en", log=quiet) == []
    run_batch(m, batch, str(tmp_path / "out"), lang="en", qc_retries=1,
              sleep=lambda *_: None, log=quiet)
    assert batch.sent == [["haha"], ["haha"]]
    [row] = m.iter_rows()
    assert (row["status"], row["review"], row["qc_takes"], row["attempts"]) == \
        (DONE, "ends on sound (-25 dB)", 1, 1)
    assert qc.read_pcm(tmp_path / "out" / row["file_path"]) == clip_pcm16(end_db=-25)
    m.close()
