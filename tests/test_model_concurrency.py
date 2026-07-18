"""Regression tests for the src.model synthesis lock.

OmniVoice isn't thread-safe, but src.server offloads /v1/tts synthesis to
asyncio's default (multi-threaded) executor. src.model._MODEL_LOCK must
serialize all access to the model, and critically must cover seeding +
generate() as one atomic unit — otherwise concurrent callers can interleave
torch's global RNG seed and silently generate with the wrong seed, breaking
the phrase bank's reproducibility guarantee.

Uses a fake model injected into src.model._model (the real model is far too
heavy for unit tests), following the same patching style as tests/conftest.py
(patch names on src.model directly, since src.server imports it lazily).
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

import src.model as model_mod


@pytest.fixture
def restore_model():
    """Ensure we never leak fake model state into other test modules."""
    prev_model = model_mod._model
    yield
    model_mod._model = prev_model


class _ConcurrencyProbeModel:
    """Fake model that tracks how many threads are inside .generate() at once."""

    def __init__(self):
        self.lock = threading.Lock()
        self.current = 0
        self.max_seen = 0

    def generate(self, **kwargs):
        with self.lock:
            self.current += 1
            self.max_seen = max(self.max_seen, self.current)
        time.sleep(0.02)
        with self.lock:
            self.current -= 1
        return [np.zeros(4, dtype=np.float32)]


def test_concurrent_generate_calls_serialize(restore_model):
    """8 threads hammering generate() must never overlap inside .generate()."""
    fake = _ConcurrencyProbeModel()
    model_mod._model = fake

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(model_mod.generate, text=f"hello {i}") for i in range(8)]
        for f in futures:
            f.result()

    assert fake.max_seen == 1, (
        f"expected generate() calls to serialize (max concurrency 1), "
        f"observed {fake.max_seen} concurrent calls"
    )


class _SeedProbeModel:
    """Fake model that records, per call, the seed its caller passed in versus
    the torch global seed actually in effect when .generate() runs.

    The caller's intended seed is smuggled in via `text` (e.g. "case-1042")
    since src.model.generate() consumes `seed` itself (for _seed_everything)
    and only forwards text/ref_audio/etc. to _model.generate(**kwargs).
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.results = []  # list of (expected_seed, observed_seed)

    def generate(self, **kwargs):
        expected_seed = int(kwargs["text"].split("-")[1])
        # Sleep here (inside what should be the locked region) to widen the
        # window for a seed/generate interleaving race, if the lock doesn't
        # cover seeding and generation atomically.
        time.sleep(0.02)
        observed_seed = torch.initial_seed()
        with self.lock:
            self.results.append((expected_seed, observed_seed))
        return [np.zeros(4, dtype=np.float32)]


def test_concurrent_generate_seeding_is_atomic(restore_model):
    """Each thread's own seed must still be in effect when its generate runs.

    Regression test for the interleaving bug: seed(A) -> seed(B) -> generate
    under A's request, silently using B's seed. If seeding isn't atomic with
    generation under the same lock, some thread will observe a seed other
    than the one it passed in.
    """
    fake = _SeedProbeModel()
    model_mod._model = fake

    seeds = list(range(1000, 1016))

    def worker(seed):
        model_mod.generate(text=f"case-{seed}", seed=seed)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, s) for s in seeds]
        for f in futures:
            f.result()

    assert len(fake.results) == len(seeds)
    for expected_seed, observed_seed in fake.results:
        assert observed_seed == expected_seed, (
            f"thread that called generate(seed={expected_seed}) observed "
            f"torch seed {observed_seed} at generate-time — seeding and "
            f"generation were not atomic, so this request silently used "
            f"another concurrent caller's seed"
        )
