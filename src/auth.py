"""JWT bearer-token auth dependency for write/synthesis routes.

Env-gated: JWT_SECRET unset -> auth is a no-op (dev / back-compat mode, keeps
existing open deployments working unchanged). Set it to require a valid
HS256 bearer token on the routes that depend on `require_auth`.
"""
import os

import jwt
from fastapi import HTTPException, Request

_UNAUTHORIZED_HEADERS = {"WWW-Authenticate": "Bearer"}


def require_auth(request: Request) -> None:
    secret = os.getenv("JWT_SECRET")
    if not secret:
        return  # auth disabled

    auth_header = request.headers.get("Authorization", "")
    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(401, "missing bearer token", headers=_UNAUTHORIZED_HEADERS)

    audience = os.getenv("JWT_AUDIENCE")
    issuer = os.getenv("JWT_ISSUER")
    options = {} if audience else {"verify_aud": False}
    try:
        jwt.decode(
            token, secret, algorithms=["HS256"],
            audience=audience, issuer=issuer, options=options,
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(401, f"invalid token: {exc}", headers=_UNAUTHORIZED_HEADERS) from exc
