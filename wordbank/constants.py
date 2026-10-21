"""Project-wide TTS sample rate: Gemini TTS and OmniVoice both emit 24 kHz / 16-bit mono PCM."""
SAMPLE_RATE = 24_000

# Served bank format (libopus). scripts/encode_wavs.py encodes the batch bank
# with these, and /v1/speak encodes the clips it adds to the bank with the same
# values, so every banked OGG has one quality/size profile.
BANK_OGG_BITRATE = "16k"
BANK_OGG_APPLICATION = "voip"
