"""Test the local OmniVoice backend (model mocked via generate_fn)."""
import numpy as np

from wordbank.backends.local import LocalOmniVoiceBackend


def test_local_backend_passes_reference_and_fixed_seed():
    seen = {}

    def fake_generate(text, ref_audio=None, ref_text=None, seed=None):
        seen["call"] = (text, ref_audio, ref_text, seed)
        return np.zeros(2400, dtype=np.float32)

    backend = LocalOmniVoiceBackend(
        ref_audio="refs/en-gb.wav", ref_text="hello there", voice="en-gb",
        seed=42, generate_fn=fake_generate,
    )
    res = backend.synthesize("cat")

    assert backend.name == "local"
    assert seen["call"] == ("cat", "refs/en-gb.wav", "hello there", 42)
    assert res.sample_rate == 24_000
    assert len(res.pcm) == 2400 and res.billed_chars == 3
    assert res.seed == 42                       # fixed seed recorded


def test_local_backend_uses_random_seed_when_unset():
    seeds = []

    def fake_generate(text, ref_audio=None, ref_text=None, seed=None):
        seeds.append(seed)
        return np.zeros(10, dtype=np.float32)

    backend = LocalOmniVoiceBackend(generate_fn=fake_generate)  # no fixed seed
    r1 = backend.synthesize("a")
    r2 = backend.synthesize("b")

    # A concrete seed is chosen per call, passed to the model, and returned.
    assert isinstance(r1.seed, int) and r1.seed == seeds[0]
    assert isinstance(r2.seed, int) and r2.seed == seeds[1]
    assert all(0 <= s < 2 ** 31 for s in seeds)
