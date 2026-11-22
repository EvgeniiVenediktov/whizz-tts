"""Edge mode: fetch a /v1/speak miss (or a voice preview) from an upstream
whizz-tts server.

An edge (a cache deployed next to its client) keeps its own copy of the bank
and, instead of calling Gemini itself, forwards misses to the upstream
server's /v1/speak, then saves what comes back; /v1/gemini/tts previews are
passed through (nothing saved). With `cafile` (SPEAK_UPSTREAM_CA) the edge
trusts only that certificate, e.g. the upstream's own self-signed one, not the
system CAs. Stdlib only (urllib +
urlencoded form, which FastAPI's Form() accepts), so it adds nothing to the
bank-only image.
"""
import functools
import json
import socket
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional

MAX_REPLY_BYTES = 5 * 1024 * 1024       # a clip is ~3 KB/s of speech


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None             # a 3xx is an error: the bearer token never goes elsewhere


@functools.lru_cache(maxsize=None)
def _opener(cafile: Optional[str]) -> urllib.request.OpenerDirector:
    handlers = [_NoRedirect()]
    if cafile:
        handlers.append(urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=cafile)))
    return urllib.request.build_opener(*handlers)


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
    voice: Optional[str]    # upstream's X-Voice ("Puck"; from older code "gemini-Puck" for a phrase)
    clip_id: Optional[str]  # upstream's X-Clip-Id (absent from older code, and for a word)


def fetch_speech(base_url: str, token: Optional[str], text: str, lang: str,
                 voice: Optional[str] = None, instruct: str = "",
                 timeout: float = 60.0, cafile: Optional[str] = None) -> UpstreamSpeech:
    """POST text to `<base_url>/v1/speak` in audio mode and return the OGG."""
    fields = {"text": text, "lang": lang, "voice": voice, "instruct": instruct}
    body, headers = _post(base_url, "/v1/speak", token, fields, "audio/ogg", timeout, cafile)
    if body[:4] != b"OggS":
        raise UpstreamError(502, f"upstream returned {headers.get('Content-Type')!r}, not OGG audio")
    return UpstreamSpeech(ogg=body, voice=headers.get("X-Voice"), clip_id=headers.get("X-Clip-Id"))


def fetch_preview(base_url: str, token: Optional[str], fields: dict,
                  timeout: float = 60.0, cafile: Optional[str] = None) -> bytes:
    """POST a voice preview (/v1/gemini/tts form fields) upstream -> the WAV."""
    body, headers = _post(base_url, "/v1/gemini/tts", token, fields, "audio/wav", timeout, cafile)
    if body[:4] != b"RIFF":
        raise UpstreamError(502, f"upstream returned {headers.get('Content-Type')!r}, not WAV audio")
    return body


def _post(base_url: str, path: str, token: Optional[str], fields: dict, accept: str,
          timeout: float, cafile: Optional[str]) -> tuple:
    """POST the non-empty `fields` urlencoded -> (body, headers).

    Upstream 400s are passed through (the request itself is bad); every other
    failure is the edge's upstream problem, not the caller's: 502 (504 on a
    timeout), with upstream's detail in the message. Redirects are not
    followed, and a reply over MAX_REPLY_BYTES is refused.
    """
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=urllib.parse.urlencode({k: v for k, v in fields.items() if v}).encode(),
        headers={"Accept": accept, **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST",
    )
    try:
        with _opener(cafile).open(req, timeout=timeout) as resp:
            body = resp.read(MAX_REPLY_BYTES + 1)
            headers = resp.headers
    except urllib.error.HTTPError as exc:
        detail = _detail(exc.read(4096)) or exc.reason
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
        if isinstance(exc.reason, ssl.SSLCertVerificationError):
            raise UpstreamError(502, f"upstream TLS certificate not trusted "
                                     f"({exc.reason.verify_message})") from exc
        raise UpstreamError(502, "upstream unreachable") from exc

    if len(body) > MAX_REPLY_BYTES:
        raise UpstreamError(502, f"upstream reply exceeds {MAX_REPLY_BYTES} bytes")
    return body, headers


def _detail(raw: bytes) -> str:
    """FastAPI's {"detail": ...} body, else the raw text (truncated)."""
    try:
        return str(json.loads(raw)["detail"])
    except (ValueError, KeyError, TypeError):
        return raw[:300].decode(errors="replace")
