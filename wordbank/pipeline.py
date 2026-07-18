"""Sync generation pipeline: pull pending → synthesize → post-process →
(optionally verify) → write a WAV master. Backend-agnostic (see backends/)."""
import os
import pathlib
from typing import Optional

from . import postproc
from .backends.base import TTSBackend
from .manifest import FAILED, QUARANTINED, VERIFIED, Manifest
from .paths import safe_filename
from .ratelimit import RateLimiter
from .verify import Verifier


class VerificationError(Exception):
    pass


class WordBankPipeline:
    def __init__(self, manifest: Manifest, backend: TTSBackend,
                 verifier: Optional[Verifier], out_dir: str, lang: str = "en",
                 voice: Optional[str] = None, max_attempts: int = 3,
                 rate_limiter: Optional[RateLimiter] = None):
        self.manifest = manifest
        self.backend = backend
        self.verifier = verifier          # None → skip STT verification
        self.out_dir = pathlib.Path(out_dir)
        self.lang = lang
        self.voice = voice or getattr(backend, "_voice", "default")
        self.max_attempts = max_attempts
        self.rate_limiter = rate_limiter

    def _rel_path(self, row_id: int, text: str) -> str:
        """WAV path relative to the bank root (out_dir) — stored in the manifest,
        resolved at serve time under WAV_ROOT/OGG_ROOT (so it's portable)."""
        shard = row_id // 1000
        return os.path.join(self.lang, f"{self.backend.name}-{self.voice}",
                            str(shard), f"{row_id}_{safe_filename(text)}.wav")

    def process_row(self, row) -> str:
        row_id, text = row["id"], row[self.manifest.text_col]
        attempts = self.manifest.start_attempt(row_id)
        transcript = None
        try:
            if self.rate_limiter is not None:
                self.rate_limiter.acquire()
            res = self.backend.synthesize(
                text, voice=self.voice, language=self.lang
            )
            pcm = postproc.peak_normalize(
                postproc.trim_silence(res.pcm, res.sample_rate)
            )
            if self.verifier is not None:
                ok, transcript = self.verifier.verify(pcm, res.sample_rate, text)
                if not ok:
                    raise VerificationError(f"STT mismatch: heard {transcript!r}")

            rel = self._rel_path(row_id, text)
            path = self.out_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            postproc.write_wav_from_array(path, pcm, res.sample_rate)
            # Record provenance: the seed and the clone reference used, so every
            # generated clip is reproducible.
            self.manifest.set_status(
                row_id, VERIFIED, file_path=rel, transcript=transcript,
                billed_chars=res.billed_chars, voice=self.voice, seed=res.seed,
                ref_audio=getattr(self.backend, "ref_audio", None),
                ref_text=getattr(self.backend, "ref_text", None),
            )
            return VERIFIED
        except Exception as exc:
            status = QUARANTINED if attempts >= self.max_attempts else FAILED
            self.manifest.set_status(
                row_id, status, error=str(exc)[:500], transcript=transcript
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
