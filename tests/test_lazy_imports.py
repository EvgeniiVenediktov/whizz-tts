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
