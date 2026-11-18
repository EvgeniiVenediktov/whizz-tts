"""Guards the lazy-import invariant: src/server.py imports google-genai and
src.audio inside handlers, and the image ships requirements-base.txt only (no
numpy/soundfile). An eager top-level import of a dev-only package breaks that
image on startup.

These run in a SUBPROCESS: other tests import those modules first, so an
in-process check of sys.modules would pass vacuously.
"""
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).parent.parent

_CHECK_SCRIPT = """
import sys
import src.server
forbidden = ("google.genai", "numpy", "soundfile")
leaked = [m for m in forbidden if m in sys.modules]
print("LEAKED:", leaked)
sys.exit(1 if leaked else 0)
"""


def test_server_import_does_not_pull_in_ml_stack():
    """Importing src.server alone (no handler called) must not import
    google.genai or numpy/soundfile."""
    result = subprocess.run(
        [sys.executable, "-c", _CHECK_SCRIPT],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "importing src.server pulled in a module that should only be "
        "imported lazily inside handlers (breaks the base-requirements "
        f"image):\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


_BANK_ONLY_SPEAK = """
import os, sys, tempfile
for mod in ("numpy", "soundfile"):
    sys.modules[mod] = None          # absent from the image
d = tempfile.mkdtemp()
os.environ.update(BANK_DB=d + "/wb.db", OGG_ROOT=d + "/ogg", WAV_ROOT=d + "/wav")
import wordbank.gemini_tts as g
g.synthesize_pcm = lambda text, voice="Kore", instruct="", model=None, **kw: b"\\x00\\x10" * 12000
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
    """The image ships base requirements only (no numpy/soundfile), and /v1/speak must still generate, encode, save, and re-serve a
    clip there. Subprocess for the same reason as above: in-process, those
    modules are already loaded by other tests."""
    result = subprocess.run(
        [sys.executable, "-c", _BANK_ONLY_SPEAK],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        "/v1/speak needs a module the image doesn't have:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr[-2000:]}"
    )
