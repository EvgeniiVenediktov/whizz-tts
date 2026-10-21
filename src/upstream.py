"""Edge mode: fetch a /v1/speak miss from an upstream whizz-tts server.

An edge (a cache deployed next to its client) keeps its own copy of the bank
and, instead of calling Gemini itself, forwards misses to the upstream
server's /v1/speak, then saves what comes back. Stdlib only (urllib +
urlencoded form, which FastAPI's Form() accepts), so it adds nothing to the
bank-only image.
"""
import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional


class UpstreamError(Exception):
    """Upstream call failed. `status` is the HTTP status the edge should
    answer with; `detail` says what went wrong upstream."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


@dataclass
class UpstreamSpeech:
    ogg: bytes
    voice: Optional[str]    # upstream's X-Voice label (e.g. "gemini-Puck", or "Kore" for a word)
    source: Optional[str]   # upstream's X-Source ("bank" | "gemini")


def fetch_speech(base_url: str, token: Optional[str], text: str, lang: str,
                 voice: Optional[str] = None, instruct: str = "",
                 timeout: float = 60.0) -> UpstreamSpeech:
    """POST text to `<base_url>/v1/speak` and return the OGG it answers with.

    Upstream 400s are passed through (the request itself is bad); every other
    failure is the edge's upstream problem, not the caller's: 502 (504 on a
    timeout), with upstream's detail in the message.
    """
    fields = {"text": text, "lang": lang}
    if voice:
        fields["voice"] = voice
    if instruct:
        fields["instruct"] = instruct
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/speak",
        data=urllib.parse.urlencode(fields).encode(),
        headers={"Authorization": f"Bearer {token}"} if token else {},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            headers = resp.headers
    except urllib.error.HTTPError as exc:
        detail = _detail(exc.read()) or exc.reason
        if exc.code == 400:
            raise UpstreamError(400, detail) from exc
        if exc.code in (401, 403):
            raise UpstreamError(502, f"upstream rejected the edge's token ({exc.code}): {detail}") from exc
        raise UpstreamError(502, f"upstream {exc.code}: {detail}") from exc
    except (socket.timeout, TimeoutError) as exc:
        raise UpstreamError(504, f"upstream timed out after {timeout:g}s") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (socket.timeout, TimeoutError)):
            raise UpstreamError(504, f"upstream timed out after {timeout:g}s") from exc
        raise UpstreamError(502, f"upstream unreachable: {exc.reason}") from exc

    if body[:4] != b"OggS":
        raise UpstreamError(502, f"upstream returned {headers.get('Content-Type')!r}, not OGG audio")
    return UpstreamSpeech(ogg=body, voice=headers.get("X-Voice"), source=headers.get("X-Source"))


def _detail(raw: bytes) -> str:
    """FastAPI's {"detail": ...} body, else the raw text (truncated)."""
    try:
        return str(json.loads(raw)["detail"])
    except (ValueError, KeyError, TypeError):
        return raw[:300].decode(errors="replace")
