"""Guards the bank-only deployment invariant from AGENTS.md:

    "src/server.py imports src.model / src.audio lazily, inside handlers —
    not at top level. That's what lets the bank-only image ship without
    torch/omnivoice. Don't hoist them."

A bank-only image serves the pre-generated word/phrase bank only (no live
synthesis) and is built without torch, omnivoice, or google-genai to keep the
image small. If src/server.py — or anything it imports at module level — ever
grows an eager `import torch` / `import omnivoice` / `import google.genai`,
that image breaks on startup, because those packages simply aren't installed
there. AGENTS.md notes this cost real debugging time; nothing enforced it.

This has to run in a SUBPROCESS, not in-process. torch and omnivoice are
installed in this dev venv, and other tests in the suite import them before
this test runs, so by the time an in-process check executed, those modules
could already be sitting in sys.modules regardless of whether src.server
itself ever imports them — the assertion would pass vacuously no matter what
src/server.py does. Spawning a fresh interpreter guarantees an empty
sys.modules, so the only way a forbidden module shows up after `import
src.server` is if importing src.server actually pulled it in.
"""
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).parent.parent

_CHECK_SCRIPT = """
import sys
import src.server
forbidden = ("torch", "omnivoice", "google.genai")
leaked = [m for m in forbidden if m in sys.modules]
print("LEAKED:", leaked)
sys.exit(1 if leaked else 0)
"""


def test_server_import_does_not_pull_in_ml_stack():
    """Importing src.server alone (no handler called) must not import torch,
    omnivoice, or google.genai — they belong behind the lazy imports inside
    the synth/model handlers, per the AGENTS.md invariant quoted above."""
    result = subprocess.run(
        [sys.executable, "-c", _CHECK_SCRIPT],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "importing src.server pulled in a module that should only be "
        "imported lazily inside handlers (breaks the bank-only, "
        f"no-ML-stack image):\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


_BANK_ONLY_SPEAK = """
import os, sys, tempfile
for mod in ("numpy", "soundfile", "librosa", "torch", "omnivoice"):
    sys.modules[mod] = None          # absent from the bank-only image
d = tempfile.mkdtemp()
os.environ.update(BANK_DB=d + "/wb.db", OGG_ROOT=d + "/ogg", WAV_ROOT=d + "/wav")
import wordbank.gemini_tts as g
g.synthesize_pcm = lambda text, voice="Kore", instruct="", model=None: b"\\x00\\x10" * 12000
from fastapi.testclient import TestClient
from src.server import app
with TestClient(app) as c:
    miss = c.post("/v1/speak", data={"text": "hello there", "voice": "Puck"})
    hit = c.post("/v1/speak", data={"text": "hello there", "voice": "Puck"})
got = (miss.status_code, miss.headers.get("x-source"), miss.content[:4],
       hit.status_code, hit.headers.get("x-source"))
print(got, miss.text[:300] if miss.status_code != 200 else "")
sys.exit(0 if got == (200, "gemini", b"OggS", 200, "bank") else 1)
"""


def test_speak_get_or_generate_works_without_numpy_or_ml_stack():
    """The bank-only image ships base requirements only (no numpy/soundfile/
    torch), and /v1/speak must still generate, encode, save, and re-serve a
    clip there. Subprocess for the same reason as above: in-process, those
    modules are already loaded by other tests."""
    result = subprocess.run(
        [sys.executable, "-c", _BANK_ONLY_SPEAK],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        "/v1/speak needs a module the bank-only image doesn't have:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr[-2000:]}"
    )
