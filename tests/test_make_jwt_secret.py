"""scripts/make_jwt_secret.py: writes JWT_SECRET into an env file safely."""
import pathlib
import stat
import subprocess
import sys

import jwt

REPO_ROOT = pathlib.Path(__file__).parent.parent
SCRIPT = REPO_ROOT / "scripts" / "make_jwt_secret.py"


def _run(env_file, *args):
    r = subprocess.run([sys.executable, str(SCRIPT), "--env-file", str(env_file), *args],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout + r.stderr


def _secret(env_file):
    """The value as the entrypoint / docker-run.sh see it: sourced by sh."""
    return subprocess.run(["sh", "-c", f'. "{env_file}" && printf %s "$JWT_SECRET"'],
                          capture_output=True, text=True, check=True).stdout


def test_adds_a_strong_secret_without_touching_other_lines(tmp_path):
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=abc123\nOTHER=x")          # no trailing newline
    env.chmod(0o664)

    out = _run(env)
    secret = _secret(env)
    assert len(secret) >= 64 and secret not in out             # never printed
    assert env.read_text().startswith("GEMINI_API_KEY=abc123\nOTHER=x\nJWT_SECRET=")
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    token = jwt.encode({"sub": "dev"}, secret, algorithm="HS256")
    assert jwt.decode(token, secret, algorithms=["HS256"])["sub"] == "dev"


def test_rerun_keeps_the_secret_and_rotate_replaces_it(tmp_path):
    env = tmp_path / ".env"
    _run(env)                                                  # creates the file
    first = _secret(env)

    assert "left unchanged" in _run(env)
    assert _secret(env) == first

    env.write_text(env.read_text() + "export JWT_SECRET=dup\nGEMINI_API_KEY=k\n")
    _run(env, "--rotate")
    rotated = _secret(env)
    assert rotated not in (first, "dup")
    text = env.read_text()
    assert text.count("JWT_SECRET=") == 1 and "GEMINI_API_KEY=k\n" in text
