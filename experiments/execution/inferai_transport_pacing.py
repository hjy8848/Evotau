"""Process-wide sliding-window pacing for a provider credential group.

This limits dispatch rate, not total budget, episode workers, or model output.
No retry, payload modification, cache access or fitness logic lives here.
"""
from collections import deque
from threading import Lock
from time import monotonic, sleep

class SlidingWindowPacer:
    def __init__(self, max_requests=30, window_seconds=60.1):
        if type(max_requests) is not int or max_requests<1 or window_seconds<=0:
            raise ValueError('pacing parameters must be positive')
        self.max_requests=max_requests
        self.window_seconds=window_seconds
        self._timestamps=deque()
        self._lock=Lock()

    def acquire(self):
        started=monotonic()
        while True:
            with self._lock:
                now=monotonic()
                while self._timestamps and now-self._timestamps[0]>=self.window_seconds:
                    self._timestamps.popleft()
                if len(self._timestamps)<self.max_requests:
                    self._timestamps.append(now)
                    return now-started
                wait=self.window_seconds-(now-self._timestamps[0])
            sleep(max(wait,0.001))
