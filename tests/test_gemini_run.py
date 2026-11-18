"""The sync Gemini runner (`wordbank gemini-run`) and `wordbank regenerate`.

run_sync gets an injected synth, clock and sleep, so rate limiting and backoff
are checked against a fake clock (no real waiting) and no request leaves the
process. One test runs the real synthesize_pcm (faked transports) to show a
3.8 model never gets an inline instruction.
"""
import json
import types
import wave

import pytest

from wordbank.gemini_run import run_sync
from wordbank.gemini_tts import GeminiAPIError, GeminiNetworkError, GeminiNoAudioError
from wordbank.manifest import DONE, FAILED, PENDING, Manifest

LITE = "gemini-3.8-flash-lite-tts"
STYLE = ("read the word exactly as written, plain and neutral, in English; "
         "no acting, laughing or translating")
WORDS = ["laughing", "haha", "russian", "calling", "moscow"]


def quiet(*_):
    pass


class FakeTime:
    """A clock that only moves when slept on; records every sleep."""
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s


class FakeSynth:
    """synthesize_pcm stand-in: calls on_request (the rate limiter) like the
    real one, records (word, clock time, kwargs), and answers from `script`
    per word (an Exception to raise, or a list consumed one call at a time);
    otherwise PCM whose length encodes the word."""
    def __init__(self, t, script=None):
        self.t, self.script, self.calls = t, script or {}, []

    def __call__(self, text, voice=None, instruct="", model=None, style=None, on_request=None):
        on_request()
        self.calls.append((text, self.t.now, dict(voice=voice, instruct=instruct,
                                                  model=model, style=style)))
        outcome = self.script.get(text)
        if isinstance(outcome, list):
            outcome = outcome.pop(0) if outcome else None
        if isinstance(outcome, BaseException):
            raise outcome
        return b"\x01\x00" * (100 + len(text))


@pytest.fixture
def bank(tmp_path):
    m = Manifest(str(tmp_path / "wb.db"))
    m.add_words(WORDS, "en-gb")
    m.add_words(["bonjour"], "fr-fr")                     # another language: not touched
    yield types.SimpleNamespace(m=m, out=tmp_path / "out", t=FakeTime())
    m.close()


def _run(bank, synth, **kw):
    kw = {"lang": "en-gb", "voice": "en-gb-tutor-11", "model": LITE, "style": STYLE,
          "clock": bank.t.clock, "sleep": bank.t.sleep, "log": quiet, **kw}
    return run_sync(bank.m, str(bank.out), synth=synth, **kw)


def _rows(bank):
    return {r["word"]: r for r in bank.m.iter_rows()}


def test_writes_the_batch_layout_with_provenance(bank):
    result = _run(bank, FakeSynth(bank.t))
    assert result == {"done": 5, "failed": 0, "stopped": None}
    rows = _rows(bank)
    for word in WORDS:
        r = rows[word]
        assert r["file_path"] == f"en-gb/gemini-en-gb-tutor-11/0/{r['id']}_{word}.wav"
        assert (r["status"], r["voice"], r["model"], r["style"], r["attempts"]) == \
            (DONE, "en-gb-tutor-11", LITE, STYLE, 1)
        with wave.open(str(bank.out / r["file_path"]), "rb") as wf:
            assert wf.getframerate() == 24_000 and wf.getnframes() == 100 + len(word)
    assert rows["bonjour"]["status"] == PENDING                    # --lang scoped


def test_requests_are_spaced_by_rpm(bank):
    synth = FakeSynth(bank.t)
    _run(bank, synth, rpm=10)
    starts = [t for _, t, _ in synth.calls]
    assert starts == pytest.approx([i * 6.0 for i in range(len(WORDS))])


def test_a_rerun_resumes_where_the_last_run_stopped(bank):
    synth = FakeSynth(bank.t, {"russian": KeyboardInterrupt()})    # killed mid-run
    with pytest.raises(KeyboardInterrupt):
        _run(bank, synth)
    assert [r["status"] for r in bank.m.iter_rows()][:3] == [DONE, DONE, PENDING]

    again = FakeSynth(bank.t)
    assert _run(bank, again)["done"] == 3
    assert [w for w, _, _ in again.calls] == ["russian", "calling", "moscow"]
    assert _run(bank, FakeSynth(bank.t)) == {"done": 0, "failed": 0, "stopped": None}


def test_a_transient_error_is_retried_with_backoff(bank):
    synth = FakeSynth(bank.t, {"haha": [GeminiAPIError("429 quota", code=429)]})
    assert _run(bank, synth)["done"] == 5
    r = _rows(bank)["haha"]
    assert (r["status"], r["attempts"], r["error"]) == (DONE, 2, None)
    assert 5.0 in bank.t.sleeps                                   # the backoff
    starts = [t for _, t, _ in synth.calls]
    assert all(b - a >= 6.0 - 1e-9 for a, b in zip(starts, starts[1:]))   # still rate-limited


def test_persistent_transient_errors_stop_the_run_at_max_attempts(bank):
    """No response at all counts as transient too; a row out of attempts is
    skipped by later runs."""
    synth = FakeSynth(bank.t, {"haha": GeminiNetworkError("unreachable")})
    result = _run(bank, synth, max_attempts=3)
    assert result["stopped"] and "re-run later" in result["stopped"]
    rows = _rows(bank)
    assert (rows["haha"]["status"], rows["haha"]["attempts"]) == (FAILED, 3)
    assert (rows["russian"]["status"], rows["russian"]["attempts"]) == (PENDING, 0)
    assert _run(bank, FakeSynth(bank.t))["done"] == 3             # haha is out of attempts


def test_a_permanent_error_fails_the_row_and_moves_on(bank):
    synth = FakeSynth(bank.t, {"haha": GeminiNoAudioError("no audio (status=failed)")})
    assert _run(bank, synth) == {"done": 4, "failed": 1, "stopped": None}
    r = _rows(bank)["haha"]
    assert (r["status"], r["attempts"]) == (FAILED, 1) and "status=failed" in r["error"]
    assert [w for w, _, _ in synth.calls].count("haha") == 1    # not retried in-run


def test_consecutive_failures_stop_the_run(bank):
    bad_key = GeminiAPIError("Gemini Interactions 403: API key not valid", code=403)
    synth = FakeSynth(bank.t, {w: bad_key for w in WORDS})
    result = _run(bank, synth, stop_after=3)
    assert result["failed"] == 3 and "3 consecutive" in result["stopped"]
    assert len(synth.calls) == 3


def test_a_voice_that_is_not_one_path_segment_is_refused(bank):
    with pytest.raises(ValueError, match="voice"):
        _run(bank, FakeSynth(bank.t), voice="../x")


def test_a_38_model_never_gets_an_inline_instruction(bank, gemini_api):
    """Through the real synthesize_pcm: the word is sent verbatim and the
    instruction travels as the speech_metadata style."""
    import wordbank.gemini_tts as gemini_tts
    _run(bank, gemini_tts.synthesize_pcm, limit=2, instruction="Say it plainly.", style=None)
    assert gemini_api.prompts == []
    assert [(b["input"][0]["content"][0]["text"],
             b["input"][0]["content"][0]["annotations"][0]["style"]) for b in gemini_api.bodies] \
        == [("laughing", "Say it plainly."), ("haha", "Say it plainly.")]
    assert _rows(bank)["laughing"]["style"] == "Say it plainly."


# --- CLI: gemini-run and regenerate ------------------------------------------------

def _cli(tmp_path, *argv):
    from wordbank.__main__ import main
    main(["--db", str(tmp_path / "wb.db"), "--out", str(tmp_path / "words"),
          "--lang", "en-gb", *argv])


def _done_bank(tmp_path, words, voice="en-gb-tutor-11"):
    """Done rows with a WAV under words/ and an OGG under words_ogg/."""
    m = Manifest(str(tmp_path / "wb.db"))
    m.add_words(words, "en-gb")
    for r in m.iter_rows():
        rel = f"en-gb/gemini-{voice}/0/{r['id']}_{r['word']}.wav"
        for root, ext, data in (("words", ".wav", b"RIFF-old-"), ("words_ogg", ".ogg", b"OggS-old-")):
            f = tmp_path / root / (rel[:-4] + ext)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(data + r["word"].encode())
        m.set_status(r["id"], DONE, file_path=rel, voice=voice, model="gemini-3.1-flash-tts-preview",
                     style="old")
    m.close()


def test_regenerate_moves_the_take_and_resets_the_rows(tmp_path, capsys):
    _done_bank(tmp_path, ["laughing", "haha", "calling"])
    m = Manifest(str(tmp_path / "wb.db"))
    rid = m.get_entry("haha", "en-gb")["id"]
    m.mark_submitted([rid], "batches/old")                  # mid-batch rows reset too
    m.set_status(rid, FAILED, error="boom", file_path=f"en-gb/gemini-en-gb-tutor-11/0/{rid}_haha.wav")
    m.close()
    words = tmp_path / "regen.txt"
    words.write_text("laughing\nhaha\n# comment\nnot-a-word\n")

    _cli(tmp_path, "regenerate", "--words-file", str(words),
         "--ogg-root", str(tmp_path / "words_ogg"))

    out = capsys.readouterr().out
    assert "not-a-word" in out and "reset 2 word(s)" in out
    [take] = (tmp_path / "takes").iterdir()
    assert take.name.endswith("Z")                              # UTC stamp
    m = Manifest(str(tmp_path / "wb.db"))
    rows = {r["word"]: r for r in m.iter_rows()}
    m.close()
    assert set(rows) == {"laughing", "haha", "calling"}         # unknown word not created
    for word in ("laughing", "haha"):
        r = rows[word]
        assert (r["status"], r["attempts"], r["error"], r["batch_id"]) == (PENDING, 0, None, None)
        name = f"{r['id']}_{word}"
        assert (take / "wav" / f"{name}.wav").read_bytes() == b"RIFF-old-" + word.encode()
        assert (take / "ogg" / f"{name}.ogg").read_bytes() == b"OggS-old-" + word.encode()
        assert not (tmp_path / "words" / r["file_path"]).exists()   # moved, so encode re-runs
    assert rows["calling"]["status"] == DONE
    assert (tmp_path / "words_ogg" / (rows["calling"]["file_path"][:-4] + ".ogg")).exists()

    listing = json.loads((take / "words.json").read_text())
    assert [(e["word"], e["wav"], e["ogg"], e["model"], e["style"]) for e in listing] == [
        (w, f"wav/{rows[w]['id']}_{w}.wav", f"ogg/{rows[w]['id']}_{w}.ogg",
         "gemini-3.1-flash-tts-preview", "old") for w in ("laughing", "haha")]


def test_regenerate_then_gemini_run_makes_a_new_take(tmp_path, gemini_api):
    _done_bank(tmp_path, ["laughing", "haha", "calling"])
    words = tmp_path / "regen.txt"
    words.write_text("laughing\nhaha\n")
    _cli(tmp_path, "regenerate", "--words-file", str(words))
    _cli(tmp_path, "gemini-run", "--voice", "en-gb-tutor-11", "--model", LITE,
         "--style", STYLE, "--rpm", "6000")

    assert [b["input"][0]["content"][0]["text"] for b in gemini_api.bodies] == ["laughing", "haha"]
    m = Manifest(str(tmp_path / "wb.db"))
    rows = {r["word"]: r for r in m.iter_rows()}
    m.close()
    for word in ("laughing", "haha"):
        r = rows[word]
        assert (r["status"], r["model"], r["style"]) == (DONE, LITE, STYLE)
        assert (tmp_path / "words" / r["file_path"]).read_bytes()[:4] == b"RIFF"
    assert (rows["calling"]["model"], rows["calling"]["style"]) == \
        ("gemini-3.1-flash-tts-preview", "old")


def test_gemini_run_on_a_38_model_needs_an_explicit_direction(tmp_path, gemini_api):
    """No --style/--instruction on 3.8: refused before any request (the batch
    default VOICE_INSTRUCTION would be read aloud)."""
    _done_bank(tmp_path, ["laughing"])
    m = Manifest(str(tmp_path / "wb.db"))
    m.reset([r["id"] for r in m.iter_rows()])
    m.close()
    with pytest.raises(SystemExit) as exc:
        _cli(tmp_path, "gemini-run", "--voice", "Kore", "--model", LITE)
    assert "reads inline instructions aloud" in str(exc.value.code)
    assert gemini_api.bodies == [] and gemini_api.prompts == []


def test_export_includes_provenance(tmp_path):
    _done_bank(tmp_path, ["laughing"])
    out = tmp_path / "export.csv"
    _cli(tmp_path, "export", "--output", str(out))
    header, row = out.read_text().splitlines()
    assert header.endswith(",model,style")
    assert row.endswith(",gemini-3.1-flash-tts-preview,old")
