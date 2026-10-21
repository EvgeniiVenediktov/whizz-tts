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
_LINE = re.compile(rf"^\s*(export\s+)?{KEY}\s*=")


def write_secret(env_file: pathlib.Path, rotate: bool = False) -> str:
    """Add (or with rotate=True, replace) JWT_SECRET in env_file. Returns
    "added", "rotated", or "kept" (present and rotate=False: file untouched)."""
    lines = env_file.read_text().splitlines(keepends=True) if env_file.exists() else []
    found = [i for i, line in enumerate(lines) if _LINE.match(line)]
    if found and not rotate:
        return "kept"

    new_line = f"{KEY}={secrets.token_urlsafe(64)}\n"
    if found:
        lines[found[0]] = new_line
        lines = [line for i, line in enumerate(lines) if i not in found[1:]]
    else:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(new_line)

    fd, tmp = tempfile.mkstemp(dir=env_file.parent, prefix=f".{env_file.name}.")
    try:
        with os.fdopen(fd, "w") as f:   # mkstemp creates it 0600
            f.writelines(lines)
        os.replace(tmp, env_file)
    except BaseException:
        os.unlink(tmp)
        raise
    return "rotated" if found else "added"


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
