"""wordbank/qc.py (the end-of-clip check and the best-take rule) and the
`wordbank qc` command: check a bank's clips, record the flags, and with
--retry re-take flagged words over the faked Gemini transports."""
import csv

import pytest

from tests.conftest import clip_pcm16, interaction_resp
from tests.test_wordbank_batch import ScriptedBatch
from wordbank import qc
from wordbank.gemini_batch import write_wav_from_pcm_bytes
from wordbank.gemini_tts import pcm_to_wav_bytes, route_direction
from wordbank.manifest import DONE, RETAKE, Manifest

LITE = "gemini-3.8-flash-lite-tts"
VOICE = "en-gb-tutor-11"


def test_check_flags_a_loud_end_and_implausible_lengths(tmp_path):
    cases = {"clean": (clip_pcm16(0.5), []),
             "cut-off": (clip_pcm16(0.5, end_db=-30), ["ends on sound (-30 dB)"]),
             "long": (clip_pcm16(3.5), ["too long (3.50 s)"]),
             "short": (clip_pcm16(0.1), ["too short (0.15 s)"])}
    for name, (pcm, reasons) in cases.items():
        path = tmp_path / f"{name}.wav"
        write_wav_from_pcm_bytes(path, pcm)
        check = qc.check_wav(path)
        assert check == qc.check_pcm(pcm), name            # header + tail == in memory
        assert check.reasons == reasons, name
        assert check.review == ("; ".join(reasons) or None)


def test_best_take_prefers_a_plausible_length_then_the_quieter_end():
    loud, quieter, long_but_silent = (qc.check_pcm(clip_pcm16(s, db))
                                      for s, db in ((0.5, -20), (0.5, -35), (3.5, None)))
    assert qc.best_take(("loud", loud), ("long", long_but_silent), ("quieter", quieter))[0] \
        == "quieter"
    assert qc.best_take(("long", long_but_silent), ("loud", loud))[0] == "loud"


def _bank(tmp_path, clips, model="gemini-3.1-flash-tts-preview", style="old"):
    """Done en-gb rows with real WAV masters and OGG copies: {word: pcm}."""
    m = Manifest(str(tmp_path / "wb.db"))
    m.add_words(list(clips), "en-gb", voice=VOICE)
    for r in m.iter_rows():
        rel = f"en-gb/gemini-{VOICE}/0/{r['id']}_{r['word']}.wav"
        write_wav_from_pcm_bytes(_mkdir(tmp_path / "words" / rel), clips[r["word"]])
        _mkdir(tmp_path / "words_ogg" / (rel[:-4] + ".ogg")).write_bytes(b"OggS-old")
        m.set_status(r["id"], DONE, file_path=rel, voice=VOICE, model=model, style=style)
    m.close()


def _mkdir(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _cli(tmp_path, *argv):
    from wordbank.__main__ import main
    main(["--db", str(tmp_path / "wb.db"), "--out", str(tmp_path / "words"),
          "--lang", "en-gb", *argv])


def _rows(tmp_path):
    m = Manifest(str(tmp_path / "wb.db"))
    rows = {r["word"]: dict(r) for r in m.iter_rows()}
    m.close()
    return rows


def test_qc_records_flags_and_retry_keeps_the_best_take(tmp_path, gemini_api, capsys):
    original = {"cat": clip_pcm16(), "haha": clip_pcm16(end_db=-25),
                "russian": clip_pcm16(end_db=-30)}
    _bank(tmp_path, original)
    review = tmp_path / "review.csv"

    _cli(tmp_path, "qc", "--csv", str(review))               # check only: nothing sent
    assert "checked 3 clip(s) (en-gb, every voice): 2 flagged, 2 still flagged" \
        in capsys.readouterr().out
    rows = _rows(tmp_path)
    assert (rows["cat"]["review"], rows["haha"]["review"]) == (None, "ends on sound (-25 dB)")
    assert rows["cat"]["qc_end_db"] < -40 and gemini_api.bodies == []
    assert [r["word"] for r in csv.DictReader(review.open())] == ["haha", "russian"]

    # haha's re-take passes; russian's ends louder (-20 dB) than its original
    gemini_api.interactions = [interaction_resp(pcm_to_wav_bytes(clip_pcm16(seed=1))),
                               interaction_resp(pcm_to_wav_bytes(clip_pcm16(end_db=-20)))]
    _cli(tmp_path, "qc", "--retry", "--sync", "--model", LITE, "--style", "plain", "--qc-retries", "1",
         "--rpm", "6000", "--ogg-root", str(tmp_path / "words_ogg"), "--csv", str(review))
    out = capsys.readouterr().out
    assert "2 flagged, 1 resolved by a re-take, 1 still flagged" in out
    assert [b["input"][0]["content"][0]["text"] for b in gemini_api.bodies] == ["haha", "russian"]

    rows = _rows(tmp_path)
    haha, russian = rows["haha"], rows["russian"]
    assert (haha["review"], haha["qc_takes"], haha["model"], haha["style"]) == \
        (None, 1, LITE, "plain")
    assert qc.read_pcm(tmp_path / "words" / haha["file_path"]) == clip_pcm16(seed=1)
    [take] = (tmp_path / "takes").iterdir()                 # the replaced clip, WAV + OGG
    assert qc.read_pcm(take / "wav" / f"{haha['id']}_haha.wav") == original["haha"]
    assert (take / "ogg" / f"{haha['id']}_haha.ogg").read_bytes() == b"OggS-old"
    assert (russian["review"], russian["qc_takes"], russian["model"]) == \
        ("ends on sound (-30 dB)", 1, "gemini-3.1-flash-tts-preview")
    assert qc.read_pcm(tmp_path / "words" / russian["file_path"]) == original["russian"]
    assert [(r["word"], r["reason"]) for r in csv.DictReader(review.open())] == \
        [("russian", "ends on sound (-30 dB)")]

    _cli(tmp_path, "status")
    assert "QC: 1 flagged for review, 0 never checked" in capsys.readouterr().out


def test_qc_retry_batches_the_retakes_with_each_clips_own_provenance(tmp_path, monkeypatch, capsys):
    """haha passes on its first batch re-take; russian's two are both louder at
    the end than its original, which is filed back with its flag."""
    original = {"cat": clip_pcm16(), "haha": clip_pcm16(end_db=-25),
                "russian": clip_pcm16(end_db=-30)}
    _bank(tmp_path, original, model=LITE, style="plain")
    batch = ScriptedBatch({"haha": [clip_pcm16(seed=1)],
                           "russian": [clip_pcm16(end_db=-20), clip_pcm16(end_db=-22)]})
    made = []

    def gemini_batch(model, voice, instruction, style):
        batch.model, batch.voice = model, voice
        batch.style, batch.instruction = route_direction(model, instruction, style)
        made.append((model, voice, batch.direction))
        return batch
    monkeypatch.setattr("wordbank.__main__.GeminiBatch", gemini_batch)

    _cli(tmp_path, "qc", "--retry", "--ogg-root", str(tmp_path / "words_ogg"))
    assert "2 flagged, 1 resolved by a re-take, 1 still flagged" in capsys.readouterr().out
    assert made == [(LITE, VOICE, "plain")]
    assert batch.sent == [["haha", "russian"], ["russian"]]
    rows = _rows(tmp_path)
    assert [(rows[w]["status"], rows[w]["review"], rows[w]["qc_takes"], rows[w]["attempts"],
             rows[w]["model"], rows[w]["style"]) for w in ("haha", "russian")] == [
        (DONE, None, 1, 0, LITE, "plain"), (DONE, "ends on sound (-30 dB)", 2, 0, LITE, "plain")]
    words = tmp_path / "words"
    assert qc.read_pcm(words / rows["haha"]["file_path"]) == clip_pcm16(seed=1)
    assert qc.read_pcm(words / rows["russian"]["file_path"]) == original["russian"]
    assert not list((tmp_path / "takes" / "qc").rglob("*.wav"))      # held takes filed
    assert sorted(p.name for p in (tmp_path / "takes").rglob("*_*.*")) == [
        f"{rows['haha']['id']}_haha.ogg", f"{rows['haha']['id']}_haha.wav",
        f"{rows['russian']['id']}_russian.ogg", f"{rows['russian']['id']}_russian.wav"]

    with pytest.raises(SystemExit, match="add --sync"):
        _cli(tmp_path, "qc", "--retry", "--model", LITE)


def test_accepted_clips_are_neither_reflagged_nor_retaken(tmp_path, gemini_api, capsys):
    """A person listened and kept a flagged clip: qc leaves it alone until a
    regenerate gives the word a new take."""
    _bank(tmp_path, {"tom": clip_pcm16(end_db=-34), "haha": clip_pcm16(end_db=-25)})
    keep = tmp_path / "keep.txt"
    keep.write_text("tom\n")
    gemini_api.interactions = [interaction_resp(pcm_to_wav_bytes(clip_pcm16(seed=1)))]
    _cli(tmp_path, "qc", "--accept-file", str(keep), "--retry", "--sync", "--model", LITE,
         "--style", "plain", "--rpm", "6000")
    out = capsys.readouterr().out
    assert "accepted 1 clip(s)" in out and "checked 1 clip(s)" in out
    assert [b["input"][0]["content"][0]["text"] for b in gemini_api.bodies] == ["haha"]
    tom = _rows(tmp_path)["tom"]
    assert (tom["review"], tom["qc_accepted"]) == (None, 1)

    _cli(tmp_path, "qc")                                     # a later check skips it too
    assert "checked 1 clip(s)" in capsys.readouterr().out and _rows(tmp_path)["tom"]["review"] is None

    m = Manifest(str(tmp_path / "wb.db"))
    m.reset([tom["id"]])
    m.close()
    assert _rows(tmp_path)["tom"]["qc_accepted"] == 0


def test_held_takes_of_two_banks_in_one_directory_do_not_collide(tmp_path):
    assert qc.retake_path(str(tmp_path / "a.db"), 1) != qc.retake_path(str(tmp_path / "b.db"), 1)


def test_accepting_a_clip_that_waits_for_a_retake_is_refused(tmp_path):
    """Its take is held outside the bank: accepting it would leave it unserved
    and still re-taken by the next batch."""
    _bank(tmp_path, {"haha": clip_pcm16(end_db=-25)})
    m = Manifest(str(tmp_path / "wb.db"))
    [row] = m.iter_rows()
    m.set_status(row["id"], RETAKE, file_path=row["file_path"], check=qc.check_pcm(clip_pcm16()))
    m.close()
    keep = tmp_path / "keep.txt"
    keep.write_text("haha\n")
    with pytest.raises(SystemExit, match="haha wait.s. for a QC re-take"):
        _cli(tmp_path, "qc", "--accept-file", str(keep))
    assert _rows(tmp_path)["haha"]["qc_accepted"] == 0
