"""gemini_tts.pcm_from_payload: a Gemini audio payload -> raw 16-bit mono PCM.

gemini-3.1-flash-tts-preview sends bare PCM; gemini-3.8-flash-lite-tts a whole
WAV file (fmt, data, then a 7655-byte C2PA provenance chunk). Played as PCM,
that WAV's header is a click and its C2PA chunk ~0.15 s of noise at the end of
every clip, so only the data chunk may survive. Pure — no google-genai needed.
The WAV builders here are shared with test_gemini_tts / test_wordbank_batch.
"""
import struct

import pytest

from wordbank.gemini_tts import GeminiAudioFormatError, pcm_from_payload

PCM = bytes(range(256)) * 90        # distinct byte values, so a mis-slice shows

# Shaped like the real one: odd size (so a pad byte follows), IPTC URL at the tail.
C2PA = (b"C2PA", b"cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
        .rjust(7655, b"\xa5"))


def fmt_chunk(rate=24_000, channels=1, bits=16, tag=1):
    block = channels * bits // 8
    return b"fmt ", struct.pack("<HHIIHH", tag, channels, rate, rate * block, block, bits)


def wav_payload(*chunks, riff_size=None):
    """A RIFF/WAVE file from (id, body) chunks, odd bodies padded like a real
    writer; `riff_size` overrides the (untrusted) RIFF length field."""
    body = b"WAVE" + b"".join(cid + struct.pack("<I", len(data)) + data + b"\0" * (len(data) & 1)
                              for cid, data in chunks)
    return b"RIFF" + struct.pack("<I", len(body) if riff_size is None else riff_size) + body


def gemini_lite_wav(pcm, **fmt):
    """What gemini-3.8-flash-lite-tts returns: fmt, data, then C2PA."""
    return wav_payload(fmt_chunk(**fmt), (b"data", pcm), C2PA)


def test_lite_wav_yields_only_its_data_chunk():
    assert pcm_from_payload(gemini_lite_wav(PCM), "audio/wav") == PCM


@pytest.mark.parametrize("mime", [None, "audio/L16;codec=pcm;rate=24000"])
def test_raw_pcm_passes_through_unchanged(mime):
    assert pcm_from_payload(PCM, mime) == PCM


def test_odd_sized_chunk_before_data_is_skipped_with_its_pad_byte():
    wav = wav_payload(fmt_chunk(), (b"LIST", b"INFOabc"), (b"data", PCM), C2PA)
    assert pcm_from_payload(wav) == PCM


def test_size_fields_are_not_trusted():
    assert pcm_from_payload(wav_payload(fmt_chunk(), (b"data", PCM), riff_size=0)) == PCM
    # streamed WAV: unknown lengths written as 0xFFFFFFFF -> data runs to the end
    cid, body = fmt_chunk()
    streamed = (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
                + cid + struct.pack("<I", len(body)) + body
                + b"data" + struct.pack("<I", 0xFFFFFFFF) + PCM)
    assert pcm_from_payload(streamed) == PCM


@pytest.mark.parametrize("fmt, named", [
    (dict(rate=16_000), "16000 Hz"),
    (dict(channels=2), "2 ch"),
    (dict(bits=8), "8-bit"),
    (dict(tag=3, bits=32), "format 3"),            # IEEE float
])
def test_wav_in_another_format_raises_naming_it(fmt, named):
    with pytest.raises(GeminiAudioFormatError, match=named):
        pcm_from_payload(gemini_lite_wav(PCM, **fmt))


@pytest.mark.parametrize("payload", [
    wav_payload(fmt_chunk(), C2PA),                 # no data chunk
    wav_payload((b"data", PCM), C2PA),              # no fmt chunk
    b"RIFF" + struct.pack("<I", 4 + len(PCM)) + b"AVI " + PCM,
], ids=["no-data", "no-fmt", "not-wave"])
def test_malformed_riff_raises(payload):
    with pytest.raises(GeminiAudioFormatError):
        pcm_from_payload(payload)


@pytest.mark.parametrize("mime", ["audio/L16;codec=pcm;rate=16000", "audio/L16;rate=fast"])
def test_mime_rate_other_than_24k_raises(mime):
    with pytest.raises(GeminiAudioFormatError, match="rate"):
        pcm_from_payload(PCM, mime)
