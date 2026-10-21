"""Generate JWT_SECRET (the HS256 signing key for src/auth.py) into .env.

  python scripts/make_jwt_secret.py            # add JWT_SECRET to ./.env if absent
  python scripts/make_jwt_secret.py --rotate   # replace it (invalidates every token)
  python scripts/make_jwt_secret.py --env-file /path/to/.env

The secret comes from the OS CSPRNG (`secrets`), 64 random bytes base64url-encoded
(well over HS256's 32-byte minimum), and is never printed. Other lines in the
file are left untouched; the file is rewritten atomically and set to 0600.
"""
import argparse
import os
import pathlib
import re
import secrets
import tempfile

KEY = "JWT_SECRET"


def _matches(key: str):
    return re.compile(rf"^\s*(export\s+)?{re.escape(key)}\s*=").match


def read_env(env_file: pathlib.Path) -> dict:
    """KEY=value pairs from an env file (optional `export`, optional quotes)."""
    values = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            m = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
            if m:
                values[m.group(1)] = m.group(2).strip("'\"")
    return values


def update_env(env_file: pathlib.Path, values: dict, replace: bool = True) -> dict:
    """Set KEY=value lines in env_file, leaving every other line untouched.
    With replace=False an existing key is kept as is. Returns {key: "added" |
    "replaced" | "kept"}. Rewritten atomically, mode 0600; values never logged."""
    lines = env_file.read_text().splitlines(keepends=True) if env_file.exists() else []
    result = {}
    for key, value in values.items():
        match = _matches(key)
        found = [i for i, line in enumerate(lines) if match(line)]
        if found and not replace:
            result[key] = "kept"
            continue
        new_line = f"{key}={value}\n"
        if found:
            lines[found[0]] = new_line
            lines = [line for i, line in enumerate(lines) if i not in found[1:]]
            result[key] = "replaced"
        else:
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            lines.append(new_line)
            result[key] = "added"

    fd, tmp = tempfile.mkstemp(dir=env_file.parent, prefix=f".{env_file.name}.")
    try:
        with os.fdopen(fd, "w") as f:   # mkstemp creates it 0600
            f.writelines(lines)
        os.replace(tmp, env_file)
    except BaseException:
        os.unlink(tmp)
        raise
    return result


def new_secret() -> str:
    return secrets.token_urlsafe(64)


def write_secret(env_file: pathlib.Path, rotate: bool = False) -> str:
    """Add (or with rotate=True, replace) JWT_SECRET in env_file. Returns
    "added", "rotated", or "kept" (present and rotate=False: value untouched)."""
    result = update_env(env_file, {KEY: new_secret()}, replace=rotate)[KEY]
    return {"replaced": "rotated"}.get(result, result)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env-file", default=".env", help="env file to update (default: ./.env)")
    p.add_argument("--rotate", action="store_true",
                   help="replace an existing JWT_SECRET; every issued token stops working")
    args = p.parse_args()

    env_file = pathlib.Path(args.env_file)
    result = write_secret(env_file, rotate=args.rotate)
    if result == "kept":   # idempotent: safe to run from setup scripts
        os.chmod(env_file, 0o600)
        print(f"{KEY} already set in {env_file}; left unchanged (--rotate to replace it)")
        return
    print(f"{KEY} {result} in {env_file} (mode 0600, value not shown)")
    if result == "rotated":
        print("all previously issued tokens are now invalid; mint new ones with scripts/make_token.py")


if __name__ == "__main__":
    main()
