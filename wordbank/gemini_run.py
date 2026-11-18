"""Synchronous Gemini word-bank generation (`wordbank gemini-run`): one
gemini_tts.synthesize_pcm call per pending row, so each word can carry a
`style` (Interactions speech_metadata, 3.8 models) and small sets — e.g. rows a
`regenerate` reset — finish in minutes instead of a batch job's hours.

Clips land exactly where the batch path puts them
(`<lang>/gemini-<voice>/<id // 1000>/<id>_<word>.wav`, via
gemini_batch._write_and_mark), with model/style provenance. Resumable: state
lives in the manifest, so a re-run picks up whatever is still pending or failed
under max_attempts. Requests are spaced by a RateLimiter (`rpm`, default 10 =
Tier 1), empty-audio retries inside synthesize_pcm included.

Failures:
- transient (HTTP 429 / 5xx, no response): the row is retried with backoff
  until it reaches max_attempts; if it still fails the run stops — quota or an
  outage, which more rows would only burn attempts on. Re-run later.
- anything else (400, no audio, a bad format): the row is marked failed and
  the run moves on, stopping after `stop_after` consecutive failed rows (a bad
  key, model or voice fails every row alike).
"""
import pathlib
import time
from typing import Callable, Optional

from . import gemini_tts
from .gemini_batch import _write_and_mark
from .gemini_tts import GeminiNetworkError, route_direction
from .manifest import FAILED, Manifest
from .ratelimit import RateLimiter

_TRANSIENT_BACKOFF_S = 5.0     # sleep before in-row retry n is this * 2**(n-1), capped
_TRANSIENT_BACKOFF_MAX_S = 60.0


def is_transient(exc: BaseException) -> bool:
    """HTTP 429 / 5xx (google-genai's APIError and GeminiAPIError both carry
    `.code`) or no response at all."""
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code == 429 or code >= 500
    return isinstance(exc, (GeminiNetworkError, TimeoutError, ConnectionError))


def _check_voice(voice: str) -> None:
    # the voice names a directory of every clip: one plain path segment only
    if not voice or voice in (".", "..") or any(c in voice for c in "/\\\0"):
        raise ValueError(f"unusable voice name {voice!r}")


def run_sync(manifest: Manifest, out_dir: str, *, lang: str, voice: str, model: str,
             instruction: str = "", style: Optional[str] = None, rpm: float = 10,
             max_attempts: int = 3, limit: Optional[int] = None, stop_after: int = 5,
             synth: Optional[Callable] = None, clock: Callable[[], float] = time.monotonic,
             sleep: Callable[[float], None] = time.sleep, log: Callable = print) -> dict:
    """Generate every pending `lang` row (up to `limit`) one request at a time.
    -> this run's {"done": n, "failed": n, "stopped": reason-or-None}.

    `instruction`/`style` are routed per model exactly as in synthesize_pcm
    (route_direction): a style for a model that takes none raises
    GeminiStyleUnsupportedError before any row is touched.
    """
    _check_voice(voice)
    sent_style, inline = route_direction(model, instruction, style)
    direction = sent_style or inline or None              # recorded as provenance
    synth = synth or gemini_tts.synthesize_pcm
    limiter = RateLimiter(rpm, burst=1, clock=clock, sleep=sleep)
    out = pathlib.Path(out_dir)
    rows = manifest.pending(limit=limit, max_attempts=max_attempts, lang=lang)
    how = (f"style {sent_style!r}" if sent_style
           else f"inline {inline!r}" if inline else "bare text")
    log(f"{len(rows)} row(s) to generate ({lang}, {model}, voice {voice}, {how}, {rpm:g} rpm)")
    result = {"done": 0, "failed": 0, "stopped": None}
    consecutive = 0
    for n, row in enumerate(rows, start=1):
        while True:
            attempt = manifest.start_attempt(row["id"])
            try:
                pcm = synth(row["word"], voice=voice, instruct=instruction, model=model,
                            style=style, on_request=limiter.acquire)
            except Exception as exc:  # noqa: BLE001 - classified below, recorded on the row
                manifest.set_status(row["id"], FAILED, error=str(exc)[:500])
                transient = is_transient(exc)
                if transient and attempt < max_attempts:
                    wait = min(_TRANSIENT_BACKOFF_MAX_S, _TRANSIENT_BACKOFF_S * 2 ** (attempt - 1))
                    log(f"  {row['word']!r}: {exc} - retry in {wait:g}s "
                        f"(attempt {attempt}/{max_attempts})")
                    sleep(wait)
                    continue
                result["failed"] += 1
                log(f"  {row['word']!r}: FAILED ({exc})")
                if transient:
                    result["stopped"] = (f"transient errors persist ({exc}); re-run later, "
                                         "the run resumes from the manifest")
                else:
                    consecutive += 1
                    if consecutive >= stop_after:
                        result["stopped"] = (f"{consecutive} consecutive rows failed (last: {exc}); "
                                             "check the key, model and voice")
                break
            _write_and_mark(manifest, out, lang, voice, row, pcm, model=model, style=direction)
            result["done"] += 1
            consecutive = 0
            log(f"  [{n}/{len(rows)}] {row['word']!r}: done")
            break
        if result["stopped"]:
            log(f"stopping: {result['stopped']}")
            break
    return result
