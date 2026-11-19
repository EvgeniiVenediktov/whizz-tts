"""Request spacing for the sync Gemini runner (gemini_run.py): acquire()
returns at least 60/rpm seconds after the previous one, so no 60 s window
holds more than `rpm` requests. `clock`/`sleep` are injectable for tests."""
import time
from typing import Callable, Optional


class RateLimiter:
    def __init__(self, rate_per_min: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        if rate_per_min <= 0:
            raise ValueError(f"rate_per_min must be > 0, got {rate_per_min}")
        self._interval = 60.0 / rate_per_min
        self._clock, self._sleep = clock, sleep
        self._next: Optional[float] = None

    def acquire(self) -> None:
        if self._next is not None and self._clock() < self._next:
            self._sleep(self._next - self._clock())
        self._next = self._clock() + self._interval
