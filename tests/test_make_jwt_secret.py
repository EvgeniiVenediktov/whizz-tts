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


EDGE_SCRIPT = REPO_ROOT / "scripts" / "make_edge_env.py"


def _edge_env(tmp_path, *args):
    return subprocess.run([sys.executable, str(EDGE_SCRIPT), "--upstream-env", str(tmp_path / ".env"),
                           "--env-file", str(tmp_path / ".env.edge"), *args],
                          capture_output=True, text=True)


def _sourced(env_file, key):
    return subprocess.run(["sh", "-c", f'. "{env_file}" && printf %s "${key}"'],
                          capture_output=True, text=True, check=True).stdout


def test_edge_env_gets_an_upstream_token_and_its_own_secret(tmp_path):
    _run(tmp_path / ".env")                                  # the upstream's secret
    upstream_secret = _secret(tmp_path / ".env")

    r = _edge_env(tmp_path, "--upstream", "https://tts.example.com/", "--ttl-days", "30")
    assert r.returncode == 0, r.stderr
    edge = tmp_path / ".env.edge"
    token = _sourced(edge, "SPEAK_UPSTREAM_TOKEN")
    edge_secret = _secret(edge)
    assert _sourced(edge, "SPEAK_UPSTREAM_URL") == "https://tts.example.com"
    claims = jwt.decode(token, upstream_secret, algorithms=["HS256"])   # upstream accepts it
    assert claims["sub"] == "edge" and 29 * 86400 < claims["exp"] - claims["iat"] <= 30 * 86400
    assert edge_secret != upstream_secret and len(edge_secret) >= 64
    assert token not in r.stdout and edge_secret not in r.stdout and upstream_secret not in r.stdout
    assert stat.S_IMODE(edge.stat().st_mode) == 0o600

    # re-run: fresh token (renewal), same edge secret (site tokens keep working)
    assert _edge_env(tmp_path, "--upstream", "https://tts.example.com").returncode == 0
    assert _sourced(edge, "SPEAK_UPSTREAM_TOKEN") != token
    assert _secret(edge) == edge_secret


def test_edge_env_refuses_without_an_upstream_secret(tmp_path):
    (tmp_path / ".env").write_text("GEMINI_API_KEY=k\n")
    r = _edge_env(tmp_path, "--upstream", "https://tts.example.com")
    assert r.returncode != 0 and "no JWT_SECRET" in r.stderr
    assert not (tmp_path / ".env.edge").exists()
