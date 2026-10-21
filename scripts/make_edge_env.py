"""Write an edge cache's settings into its env file (default ./.env.edge).

  python scripts/make_edge_env.py --upstream https://tts.example.com
  python scripts/make_edge_env.py --upstream https://tts.example.com --ttl-days 365 \\
      --upstream-env .env --env-file .env.edge

Sets, without printing any value (file rewritten atomically, mode 0600):
  SPEAK_UPSTREAM_URL    the upstream whizz-tts server the edge fetches misses from
  SPEAK_UPSTREAM_TOKEN  a JWT for that upstream (sub=edge), signed with the
                        upstream's JWT_SECRET read from --upstream-env; re-running
                        mints a fresh one (use it to renew before expiry)
  JWT_SECRET            the edge's OWN secret, for the site backend's tokens;
                        generated once and kept on re-runs

Then run the edge with `ENV_FILE=.env.edge scripts/docker-run.sh edge`, and mint
the site backend's token from the edge's secret:
  set -a; . ./.env.edge; set +a; python scripts/make_token.py --sub site --ttl ...
"""
import argparse
import datetime
import pathlib
import secrets
import sys
import time

import jwt

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from make_jwt_secret import new_secret, read_env, update_env  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--upstream", required=True, help="upstream base URL, e.g. https://tts.example.com")
    p.add_argument("--env-file", default=".env.edge", help="edge env file to write (default: ./.env.edge)")
    p.add_argument("--upstream-env", default=".env",
                   help="env file holding the upstream's JWT_SECRET (default: ./.env)")
    p.add_argument("--ttl-days", type=int, default=365, help="upstream token lifetime (default: 365)")
    p.add_argument("--aud", default=None, help="audience claim, if the upstream sets JWT_AUDIENCE")
    p.add_argument("--iss", default=None, help="issuer claim, if the upstream sets JWT_ISSUER")
    args = p.parse_args()

    upstream_secret = read_env(pathlib.Path(args.upstream_env)).get("JWT_SECRET")
    if not upstream_secret:
        sys.exit(f"no JWT_SECRET in {args.upstream_env}: the upstream needs one "
                 "(scripts/make_jwt_secret.py) before an edge can authenticate to it")
    if not args.upstream.startswith(("https://", "http://")):
        sys.exit("--upstream must be an http(s) URL")

    exp = int(time.time()) + args.ttl_days * 86400
    claims = {"sub": "edge", "iat": int(time.time()), "exp": exp,
              "jti": secrets.token_hex(8)}   # unique even when minted twice in a second
    if args.aud:
        claims["aud"] = args.aud
    if args.iss:
        claims["iss"] = args.iss
    token = jwt.encode(claims, upstream_secret, algorithm="HS256")

    env_file = pathlib.Path(args.env_file)
    result = update_env(env_file, {"SPEAK_UPSTREAM_URL": args.upstream.rstrip("/"),
                                   "SPEAK_UPSTREAM_TOKEN": token})
    result.update(update_env(env_file, {"JWT_SECRET": new_secret()}, replace=False))

    until = datetime.datetime.fromtimestamp(exp, datetime.timezone.utc).date()
    print(f"{env_file} (mode 0600, values not shown):")
    for key, what in result.items():
        print(f"  {key}: {what}")
    print(f"upstream token valid until {until}; re-run this script to renew it")
    if not args.upstream.startswith("https://"):
        print("warning: plain http upstream; the token and audio cross the network unencrypted")


if __name__ == "__main__":
    main()
