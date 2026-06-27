"""Orchestration: pull pending words → synthesize → post-process → verify → write."""
import pathlib
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from . import postproc
from .backends.base import TTSBackend
from .manifest import FAILED, QUARANTINED, VERIFIED, Manifest
from .paths import safe_filename
from .ratelimit import RateLimiter
from .verify import Verifier

_LANG_TO_CODE = {"en": "en-US"}


class VerificationError(Exception):
    pass


class WordBankPipeline:
    def __init__(self, manifest: Manifest, backend: TTSBackend, verifier: Verifier,
                 out_dir: str, lang: str = "en", voice: Optional[str] = None,
                 max_attempts: int = 3, rate_limiter: Optional[RateLimiter] = None):
        self.manifest = manifest
        self.backend = backend
        self.verifier = verifier
        self.out_dir = pathlib.Path(out_dir)
        self.lang = lang
        self.language_code = _LANG_TO_CODE.get(lang, lang)
        self.voice = voice or getattr(backend, "_voice", "default")
        self.max_attempts = max_attempts
        self.rate_limiter = rate_limiter

    def _word_path(self, word_id: int, word: str) -> pathlib.Path:
        shard = word_id // 1000
        d = self.out_dir / self.lang / f"{self.backend.name}-{self.voice}" / str(shard)
        return d / f"{word_id}_{safe_filename(word)}.ogg"

    def process_row(self, row) -> str:
        word_id, word = row["id"], row["word"]
        attempts = self.manifest.start_attempt(word_id)
        transcript = None
        try:
            if self.rate_limiter is not None:
                self.rate_limiter.acquire()
            res = self.backend.synthesize(
                word, voice=self.voice, language=self.language_code
            )
            pcm = postproc.peak_normalize(
                postproc.trim_silence(res.pcm, res.sample_rate)
            )
            ok, transcript = self.verifier.verify(pcm, res.sample_rate, word)
            if not ok:
                raise VerificationError(f"STT mismatch: heard {transcript!r}")

            path = self._word_path(word_id, word)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(postproc.to_ogg(pcm, res.sample_rate))
            self.manifest.set_status(
                word_id, VERIFIED, file_path=str(path),
                transcript=transcript, billed_chars=res.billed_chars,
            )
            return VERIFIED
        except Exception as exc:
            status = QUARANTINED if attempts >= self.max_attempts else FAILED
            self.manifest.set_status(
                word_id, status, error=str(exc)[:500], transcript=transcript
            )
            return status

    def run(self, limit: Optional[int] = None, chunk: int = 100) -> dict:
        """Process pending words until none remain (or `limit` processed)."""
        processed = {VERIFIED: 0, FAILED: 0, QUARANTINED: 0}
        done = 0
        while True:
            take = chunk if limit is None else min(chunk, limit - done)
            if take <= 0:
                break
            rows = self.manifest.pending(limit=take, max_attempts=self.max_attempts)
            if not rows:
                break
            for row in rows:
                processed[self.process_row(row)] += 1
                done += 1
        return processed

    def run_parallel(self, workers: int = 10, limit: Optional[int] = None) -> dict:
        """Process pending words across a thread pool.

        Pass-based: each pass takes a snapshot of pending words, processes them
        all (concurrently), then re-snapshots — so a word is never submitted
        twice while in flight. Retryable failures are picked up on the next pass.
        Rate limiting (if set) keeps the pool under the backend's RPM quota.
        """
        processed = {VERIFIED: 0, FAILED: 0, QUARANTINED: 0}
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while True:
                take = None if limit is None else (limit - done)
                if take is not None and take <= 0:
                    break
                rows = self.manifest.pending(limit=take, max_attempts=self.max_attempts)
                if not rows:
                    break
                for status in pool.map(self.process_row, rows):
                    processed[status] += 1
                    done += 1
        return processed
