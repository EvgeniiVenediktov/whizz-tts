"""Thread-safe token-bucket rate limiter.

Size it to a backend's requests-per-minute quota so parallel workers stay under
the limit instead of triggering 429s. (Local generation is sequential and needs
no limiter; this is for any future rate-limited cloud backend.)
"""
import threading
import time


class RateLimiter:
    def __init__(self, rate_per_min: float):
        self.capacity = max(1.0, float(rate_per_min))
        self._tokens = self.capacity
        self._rate = self.capacity / 60.0          # tokens per second
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a token is available, then consume one."""
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self.capacity, self._tokens + (now - self._last) * self._rate
                )
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._rate
            time.sleep(wait)
