"""Chirp 3 HD backend via Google Cloud Text-to-Speech.

Requests LINEAR16 (PCM in a WAV container) so the pipeline can post-process
before encoding to OGG. Auth: a GCP service account with the Text-to-Speech API
enabled (GOOGLE_APPLICATION_CREDENTIALS) — NOT the Gemini Developer API key.

For tests, inject `synth_fn(text, voice, language, ssml) -> wav_bytes` to avoid
needing the google-cloud library or real credentials.
"""
import io
from typing import Callable, Optional

import numpy as np
import soundfile as sf

from .base import SynthResult


class ChirpTTSBackend:
    name = "chirp"

    def __init__(
        self,
        voice: str = "en-US-Chirp3-HD-Charon",
        language: str = "en-US",
        sample_rate: int = 24_000,
        synth_fn: Optional[Callable[..., bytes]] = None,
    ):
        self._voice = voice
        self._language = language
        self._sample_rate = sample_rate
        self._synth_fn = synth_fn  # test seam; bypasses the real client
        self._client = None

    def _call_api(self, text: str, voice: str, language: str,
                  ssml: Optional[str]) -> bytes:
        from google.cloud import texttospeech  # lazy: only needed for real runs

        if self._client is None:
            self._client = texttospeech.TextToSpeechClient()
        synthesis_input = (
            texttospeech.SynthesisInput(ssml=ssml)
            if ssml else texttospeech.SynthesisInput(text=text)
        )
        voice_params = texttospeech.VoiceSelectionParams(
            language_code=language, name=voice
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=self._sample_rate,
        )
        resp = self._client.synthesize_speech(
            input=synthesis_input, voice=voice_params, audio_config=audio_config
        )
        return resp.audio_content

    def synthesize(self, text: str, *, voice: Optional[str] = None,
                   language: Optional[str] = None,
                   ssml: Optional[str] = None) -> SynthResult:
        voice = voice or self._voice
        language = language or self._language
        caller = self._synth_fn or self._call_api
        wav_bytes = caller(text, voice, language, ssml)

        pcm, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
        billed = len(ssml) if ssml else len(text)
        return SynthResult(pcm=pcm, sample_rate=sr, billed_chars=billed)
