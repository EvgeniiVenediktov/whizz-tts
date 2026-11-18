"""Thread-safe token-bucket rate limiter.

Size it to a backend's requests-per-minute quota so parallel workers stay under
the limit instead of triggering 429s. The sync Gemini runner (gemini_run.py)
uses it with burst=1: requests evenly spaced 60/rpm seconds apart, so no
60-second window ever holds more than `rpm` of them.
"""
import threading
import time
from typing import Callable, Optional


class RateLimiter:
    def __init__(self, rate_per_min: float, burst: Optional[float] = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        """`burst` is the bucket size (default: a whole minute's quota, so a
        cold start may fire `rate_per_min` requests at once). `clock`/`sleep`
        are injectable for tests."""
        if rate_per_min <= 0:
            raise ValueError(f"rate_per_min must be > 0, got {rate_per_min}")
        self.capacity = max(1.0, float(burst if burst is not None else rate_per_min))
        self._tokens = self.capacity
        self._rate = float(rate_per_min) / 60.0     # tokens per second
        self._clock = clock
        self._sleep = sleep
        self._last = clock()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a token is available, then consume one."""
        while True:
            with self._lock:
                now = self._clock()
                self._tokens = min(
                    self.capacity, self._tokens + (now - self._last) * self._rate
                )
                self._last = now
                if self._tokens >= 1.0 - 1e-9:      # float slack: a sleep of exactly `wait` counts
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._rate
            self._sleep(wait)
