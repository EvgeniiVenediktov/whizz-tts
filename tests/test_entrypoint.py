"""docker-entrypoint.sh: provision/verify the /data volume, then exec the server.

Runs the real script against a tmp "volume", with `echo STARTED` standing in
for uvicorn, so each case shows whether the container would start and why.
"""
import json
import os
import pathlib
import subprocess
import sys
import tarfile

from wordbank.library import describe
from wordbank.manifest import DONE, Manifest

REPO_ROOT = pathlib.Path(__file__).parent.parent


def _entrypoint(data, **env):
    base = {"PATH": f"{pathlib.Path(sys.executable).parent}:{os.environ['PATH']}",
            "HOME": os.environ.get("HOME", "/tmp"), "BANK_DB": str(data / "wordbank.db")}
    r = subprocess.run(["sh", "docker-entrypoint.sh", "echo", "STARTED"], cwd=REPO_ROOT,
                       env={**base, **env}, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def _add_phrase(db, ogg_root, text, voice, pid):
    m = Manifest(str(db), table="phrases", text_col="text")
    m.add_words([text], "en-us", voice=voice)
    rel = f"en-us/{voice}/phrases/0/{pid}_{text.replace(' ', '-')}.wav"
    m.set_status(m.get_entry(text, "en-us", voice)["id"], DONE, file_path=rel, voice=voice)
    m.close()
    ogg = ogg_root / (rel[:-4] + ".ogg")
    ogg.parent.mkdir(parents=True, exist_ok=True)
    ogg.write_bytes(b"OggS")


def _package(tmp_path):
    """A packaged library as scripts/pack.sh lays it out: library.json +
    tts-bank.tar.gz holding wordbank.db, output/words_ogg, output/voices."""
    src = tmp_path / "src"
    (src / "output" / "voices" / "en-us").mkdir(parents=True)
    (src / "output" / "voices" / "en-us" / "kore.wav").write_bytes(b"RIFF")
    m = Manifest(str(src / "wordbank.db"))
    m.add_words(["cat", "dog"], "en-us", voice="Kore")
    for r in m.iter_rows():
        rel = f"en-us/gemini-Kore/0/{r['id']}_{r['word']}.wav"
        m.set_status(r["id"], DONE, file_path=rel, voice="Kore")
        ogg = src / "output" / "words_ogg" / (rel[:-4] + ".ogg")
        ogg.parent.mkdir(parents=True, exist_ok=True)
        ogg.write_bytes(b"OggS")
    m.close()
    desc = describe(str(src / "wordbank.db"), str(src / "output" / "words_ogg"),
                    str(src / "output" / "voices"))
    (src / "library.json").write_text(json.dumps(desc))
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "library.json").write_text(json.dumps(desc))
    with tarfile.open(pkg / "tts-bank.tar.gz", "w:gz") as tar:
        for name in ("library.json", "wordbank.db", "output/words_ogg", "output/voices"):
            tar.add(src / name, arcname=name)
    return pkg


def _provision(pkg, data):
    data.mkdir(exist_ok=True)
    for name in ("library.json", "tts-bank.tar.gz"):
        (data / name).write_bytes((pkg / name).read_bytes())


def test_empty_volume_without_synthesis_refuses_to_start(tmp_path):
    code, out = _entrypoint(tmp_path)
    assert code == 1 and "refusing to start" in out and "STARTED" not in out


def test_empty_volume_with_gemini_key_starts_synthesis_only(tmp_path):
    code, out = _entrypoint(tmp_path, GEMINI_API_KEY="k")
    assert code == 0 and out.rstrip().endswith("STARTED")
    assert "synthesis-only" in out


def test_restart_on_runtime_built_bank_says_so(tmp_path):
    """Gemini-only volume after /v1/speak saved phrases: no description, but the
    bank is real, so the log must not claim there is no bank."""
    _add_phrase(tmp_path / "wordbank.db", tmp_path / "words_ogg", "good morning", "gemini-Puck", 1)
    code, out = _entrypoint(tmp_path, GEMINI_API_KEY="k")
    assert code == 0 and out.rstrip().endswith("STARTED")
    assert "serving the bank built at runtime" in out and "no bank" not in out


def test_archive_is_unpacked_once_and_a_grown_bank_still_verifies(tmp_path):
    data = tmp_path / "data"
    _provision(_package(tmp_path), data)

    code, out = _entrypoint(data)
    assert code == 0 and "library verified after unpack" in out
    assert (data / "words_ogg" / "en-us" / "gemini-Kore" / "0").is_dir()
    assert (data / "voices" / "en-us" / "kore.wav").exists()

    # /v1/speak adds a phrase on the live volume; the restart must not re-unpack.
    _add_phrase(data / "wordbank.db", data / "words_ogg", "good morning", "gemini-Puck", 1)
    code, out = _entrypoint(data)
    assert code == 0 and "library verified — starting" in out and "unpacking" not in out


def test_unpack_over_an_existing_bank_keeps_it_as_a_backup(tmp_path):
    """Archive added to a volume that already served /v1/speak: the old DB
    (with its phrases) is moved aside, not overwritten."""
    data = tmp_path / "data"
    data.mkdir()
    _add_phrase(data / "wordbank.db", data / "words_ogg", "good morning", "gemini-Puck", 1)
    _provision(_package(tmp_path), data)

    code, out = _entrypoint(data, GEMINI_API_KEY="k")
    assert code == 0 and "moved to" in out
    [backup] = data.glob("wordbank.db.pre-unpack-*")
    old = Manifest(str(backup), table="phrases", text_col="text")
    assert [r["text"] for r in old.iter_rows()] == ["good morning"]
    old.close()
    new = Manifest(str(data / "wordbank.db"))
    assert sorted(r["word"] for r in new.iter_rows()) == ["cat", "dog"]
    new.close()
