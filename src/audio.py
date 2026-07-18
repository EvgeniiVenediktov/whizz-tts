import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf

from wordbank.constants import SAMPLE_RATE


def pcm_to_ogg(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert a float32 PCM array to OGG/Opus bytes."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
        wav_path = wav_file.name
        sf.write(wav_path, audio, sample_rate, format="WAV", subtype="PCM_16")

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", wav_path,
                "-c:a", "libopus",
                "-b:a", "64k",
                "-f", "ogg",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    finally:
        os.unlink(wav_path)
    return result.stdout
