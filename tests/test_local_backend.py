"""Test the local OmniVoice backend (model mocked via generate_fn)."""
import numpy as np

from wordbank.backends.local import LocalOmniVoiceBackend


def test_local_backend_passes_reference_to_model():
    seen = {}

    def fake_generate(text, ref_audio=None, ref_text=None):
        seen["call"] = (text, ref_audio, ref_text)
        return np.zeros(2400, dtype=np.float32)

    backend = LocalOmniVoiceBackend(
        ref_audio="refs/en-gb.wav", ref_text="hello there", voice="en-gb",
        generate_fn=fake_generate,
    )
    res = backend.synthesize("cat")

    assert backend.name == "local"
    assert seen["call"] == ("cat", "refs/en-gb.wav", "hello there")
    assert res.sample_rate == 24_000
    assert len(res.pcm) == 2400 and res.billed_chars == 3
