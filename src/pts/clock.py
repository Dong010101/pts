"""Bookkeeping of path progress actually sent to the robot."""

from collections import deque
import math

import numpy as np


class ProgressHistory:
    def __init__(self, maxlen=20000):
        self.samples = deque(maxlen=maxlen)

    def append(self, timestamp: float, progress: float):
        if not math.isfinite(timestamp) or not math.isfinite(progress):
            raise ValueError("timestamps and progress must be finite")
        if self.samples and (timestamp < self.samples[-1][0] or progress < self.samples[-1][1] - 1e-9):
            raise ValueError("time and consumed progress must be monotonic")
        self.samples.append((timestamp, progress))

    def at(self, timestamp: float) -> float:
        if not self.samples or not math.isfinite(timestamp):
            raise ValueError("missing progress history or invalid timestamp")
        if timestamp < self.samples[0][0] and self.samples[0][1] != 0:
            raise ValueError("observation predates retained progress history")
        times, progress = np.asarray(self.samples).T
        return float(np.interp(timestamp, times, progress, left=0, right=progress[-1]))

