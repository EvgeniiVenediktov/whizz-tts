"""PCM -> OGG/Opus via ffmpeg. Stdlib only (no numpy/soundfile import), so the
bank-only image can encode /v1/speak clips with just the ffmpeg binary."""
import subprocess

from wordbank.constants import SAMPLE_RATE


def pcm16_to_ogg(pcm: bytes, sample_rate: int = SAMPLE_RATE, bitrate: str = "64k",
                 application: str = "audio") -> bytes:
    """Raw 16-bit little-endian mono PCM bytes -> OGG/Opus bytes (stdin/stdout)."""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
            "-c:a", "libopus", "-application", application, "-b:a", bitrate,
            "-f", "ogg", "pipe:1",
        ],
        input=pcm,
        capture_output=True,
        check=True,
        timeout=60,
    )
    return result.stdout

