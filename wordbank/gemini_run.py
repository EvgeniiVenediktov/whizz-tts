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

QC (wordbank/qc.py): each take is checked as it arrives; a flagged one is
re-taken on the spot (same model, voice and direction, through the same rate
limiter) up to `qc_retries` times. The first passing take is filed; if every
take is flagged, the best one is filed with its flag in `review`. Re-takes
count in qc_takes, never in the row's generation attempts; one that raises
just ends the re-takes. `best_of_takes` and `sync_session` are shared with
`wordbank qc --retry`.

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

from . import gemini_tts, qc
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


def check_voice_name(voice: str) -> None:
    # the voice names a directory of every clip: one plain path segment only
    if not voice or voice in (".", "..") or any(c in voice for c in "/\\\0"):
        raise ValueError(f"unusable voice name {voice!r}")


def sync_session(*, model: str, instruction: str = "", style: Optional[str] = None,
                 rpm: float = 10, synth: Optional[Callable] = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
    """-> (take(word, voice) -> pcm, sent style, inline prefix): one
    rate-limited synthesis per call (empty-audio retries inside synthesize_pcm
    included), directions routed for `model` exactly as synthesize_pcm does.
    A style for a model that takes none raises GeminiStyleUnsupportedError here."""
    sent_style, inline = route_direction(model, instruction, style)
    synth = synth or gemini_tts.synthesize_pcm
    limiter = RateLimiter(rpm, burst=1, clock=clock, sleep=sleep)

    def take(word: str, voice: str) -> bytes:
        return synth(word, voice=voice, instruct=instruction, model=model, style=style,
                     on_request=limiter.acquire)
    return take, sent_style, inline


def best_of_takes(word: str, first: tuple, retake: Callable[[], bytes], retries: int,
                  log: Callable = print) -> tuple:
    """`first` is a (pcm, qc.Check) take; while the best take so far is
    flagged, ask `retake()` for another, up to `retries` times, stopping at the
    first that passes. -> (pcm, Check, re-takes made) of the best take
    (qc.best_take). A re-take that raises ends the re-takes."""
    best, made = first, 0
    if best[1].flagged and retries:
        log(f"  {word!r}: flagged ({best[1].review}) - re-taking, up to {retries}")
    while best[1].flagged and made < retries:
        made += 1
        try:
            pcm = retake()
        except Exception as exc:  # noqa: BLE001 - the best take so far stands
            log(f"  {word!r}: re-take {made} failed ({exc}); keeping the best take")
            break
        new = (pcm, qc.check_pcm(pcm))
        log(f"  {word!r}: re-take {made}: {new[1].review or 'passed'}")
        best = qc.best_take(best, new)
    return best[0], best[1], made


def run_sync(manifest: Manifest, out_dir: str, *, lang: str, voice: str, model: str,
             instruction: str = "", style: Optional[str] = None, rpm: float = 10,
             max_attempts: int = 3, limit: Optional[int] = None, stop_after: int = 5,
             qc_retries: int = qc.DEFAULT_RETRIES, synth: Optional[Callable] = None,
             clock: Callable[[], float] = time.monotonic,
             sleep: Callable[[float], None] = time.sleep, log: Callable = print) -> dict:
    """Generate every pending `lang` row in `voice` (up to `limit`) one request
    at a time, re-taking QC-flagged clips up to `qc_retries` times.
    -> this run's {"done": n, "failed": n, "flagged": n (filed with a QC flag),
    "resolved": n (passed on a re-take), "stopped": reason-or-None}.

    `instruction`/`style` are routed per model exactly as in synthesize_pcm
    (route_direction): a style for a model that takes none raises
    GeminiStyleUnsupportedError before any row is touched, as does a locale
    voice on a model not verified with it (GeminiVoiceUnsupportedError).
    """
    check_voice_name(voice)
    gemini_tts.check_voice(voice, model)
    take, sent_style, inline = sync_session(model=model, instruction=instruction, style=style,
                                            rpm=rpm, synth=synth, clock=clock, sleep=sleep)
    direction = sent_style or inline or None              # recorded as provenance
    out = pathlib.Path(out_dir)
    rows = manifest.pending(limit=limit, max_attempts=max_attempts, lang=lang, voice=voice)
    how = (f"style {sent_style!r}" if sent_style
           else f"inline {inline!r}" if inline else "bare text")
    log(f"{len(rows)} row(s) to generate ({lang}, {model}, voice {voice}, {how}, {rpm:g} rpm, "
        f"up to {qc_retries} QC re-take(s))")
    result = {"done": 0, "failed": 0, "flagged": 0, "resolved": 0, "stopped": None}
    consecutive = 0
    for n, row in enumerate(rows, start=1):
        word = row["word"]
        while True:
            attempt = manifest.start_attempt(row["id"])
            try:
                pcm = take(word, voice)
            except Exception as exc:  # noqa: BLE001 - classified below, recorded on the row
                manifest.set_status(row["id"], FAILED, error=str(exc)[:500])
                transient = is_transient(exc)
                if transient and attempt < max_attempts:
                    wait = min(_TRANSIENT_BACKOFF_MAX_S, _TRANSIENT_BACKOFF_S * 2 ** (attempt - 1))
                    log(f"  {word!r}: {exc} - retry in {wait:g}s "
                        f"(attempt {attempt}/{max_attempts})")
                    sleep(wait)
                    continue
                result["failed"] += 1
                log(f"  {word!r}: FAILED ({exc})")
                if transient:
                    result["stopped"] = (f"transient errors persist ({exc}); re-run later, "
                                         "the run resumes from the manifest")
                else:
                    consecutive += 1
                    if consecutive >= stop_after:
                        result["stopped"] = (f"{consecutive} consecutive rows failed (last: {exc}); "
                                             "check the key, model and voice")
                break
            pcm, check, made = best_of_takes(word, (pcm, qc.check_pcm(pcm)),
                                             lambda: take(word, voice), qc_retries, log)
            _write_and_mark(manifest, out, lang, voice, row, pcm, model=model, style=direction,
                            check=check, qc_takes=made)
            result["done"] += 1
            consecutive = 0
            if check.flagged:
                result["flagged"] += 1
            elif made:
                result["resolved"] += 1
            log(f"  [{n}/{len(rows)}] {word!r}: done"
                + (f", flagged for review ({check.review})" if check.flagged
                   else f", passed on re-take {made}" if made else ""))
            break
        if result["stopped"]:
            log(f"stopping: {result['stopped']}")
            break
    return result
