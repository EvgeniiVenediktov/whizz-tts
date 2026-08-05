"""Mint an HS256 JWT for the write/synthesis routes (src/auth.py).

  JWT_SECRET=... python scripts/make_token.py --sub dev
  python scripts/make_token.py --secret mysecret --ttl 600 --aud my-app
"""
import argparse
import os
import sys
import time

import jwt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sub", default="dev")
    p.add_argument("--ttl", type=int, default=3600, help="seconds until expiry")
    p.add_argument("--secret", default=os.getenv("JWT_SECRET"))
    p.add_argument("--aud", default=None, help="audience claim (optional)")
    p.add_argument("--iss", default=None, help="issuer claim (optional)")
    args = p.parse_args()

    if not args.secret:
        sys.exit("no secret: pass --secret or set JWT_SECRET")

    now = int(time.time())
    claims = {"sub": args.sub, "iat": now, "exp": now + args.ttl}
    if args.aud:
        claims["aud"] = args.aud
    if args.iss:
        claims["iss"] = args.iss

    token = jwt.encode(claims, args.secret, algorithm="HS256")
    print(token)


if __name__ == "__main__":
    main()
