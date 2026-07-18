"""Test the local OmniVoice backend (model mocked via generate_fn).

The contract that matters here is REPRODUCIBILITY (see README/AGENTS.md):
every locally-generated phrase records the seed + clone reference it was
generated with, so the clip can be regenerated identically later. We verify
that contract behaviorally by injecting a `generate_fn` that is a genuine
deterministic function of everything it receives (built from
numpy.random.RandomState), rather than a spy that records call arguments.

Deriving the fake PCM from *every* forwarded input — not just the seed — is
deliberate: it means dropping any one of them (notably the clone reference,
which is where the accent comes from) changes the audio and fails a test.
"""
import zlib

import numpy as np

from wordbank.backends.local import LocalOmniVoiceBackend


def _seeded_pcm(seed, ref_audio, ref_text, text, n=2400):
    """Stand-in model: PCM is a pure function of every input the backend forwards."""
    key = f"{seed}|{ref_audio}|{ref_text}|{text}".encode()
    rng = np.random.RandomState(zlib.crc32(key))
    return rng.uniform(-1, 1, n).astype(np.float32)


def deterministic_generate(text, ref_audio=None, ref_text=None, seed=None):
    return _seeded_pcm(seed, ref_audio, ref_text, text)


def test_local_backend_fixed_seed_is_byte_identical_across_calls():
    backend = LocalOmniVoiceBackend(
        ref_audio="refs/en-gb.wav", ref_text="hello there", voice="en-gb",
        seed=42, generate_fn=deterministic_generate,
    )
    res1 = backend.synthesize("cat")
    res2 = backend.synthesize("cat")

    assert backend.name == "local"
    assert res1.sample_rate == 24_000
    assert res1.billed_chars == 3
    assert res1.seed == 42 and res2.seed == 42       # fixed seed recorded every time
    assert np.array_equal(res1.pcm, res2.pcm)         # same seed -> byte-identical clip


def test_local_backend_random_seed_varies_but_is_reproducible():
    backend = LocalOmniVoiceBackend(generate_fn=deterministic_generate)  # no fixed seed
    r1 = backend.synthesize("a")
    r2 = backend.synthesize("b")

    # No fixed seed -> a fresh seed is drawn per call, so clips differ...
    assert r1.seed != r2.seed
    assert not np.array_equal(r1.pcm, r2.pcm)

    # ...but the seed actually used is returned to the caller, so persisting
    # it and re-synthesizing with that seed reproduces the exact same clip.
    replay_backend = LocalOmniVoiceBackend(seed=r1.seed, generate_fn=deterministic_generate)
    r1_replay = replay_backend.synthesize("a")
    assert r1_replay.seed == r1.seed
    assert np.array_equal(r1.pcm, r1_replay.pcm)


def test_local_backend_forwards_clone_reference_and_text_to_model():
    """The clone reference must actually reach the model — the accent of every
    phrase comes from it. Holding the seed fixed, changing the reference (or
    its transcript, or the text) must change the audio; if the backend stopped
    forwarding one, the output would be identical and this test fails.
    """
    fixed = dict(seed=7, generate_fn=deterministic_generate)
    gb = LocalOmniVoiceBackend(ref_audio="refs/en-gb.wav", ref_text="hi", **fixed)
    us = LocalOmniVoiceBackend(ref_audio="refs/en-us.wav", ref_text="hi", **fixed)
    assert not np.array_equal(gb.synthesize("cat").pcm, us.synthesize("cat").pcm)

    other_transcript = LocalOmniVoiceBackend(
        ref_audio="refs/en-gb.wav", ref_text="a different transcript", **fixed)
    assert not np.array_equal(gb.synthesize("cat").pcm,
                              other_transcript.synthesize("cat").pcm)

    # ...and the phrase text itself reaches the model.
    assert not np.array_equal(gb.synthesize("cat").pcm, gb.synthesize("dog").pcm)
