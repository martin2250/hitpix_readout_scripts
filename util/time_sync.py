from dataclasses import dataclass, asdict
from typing import TypeVar

import numpy as np

X = TypeVar('X', int, np.ndarray)

@dataclass
class TimeSync:
    sync_counter: int  # readout time register
    sync_time: float  # time.now()
    tick_us: float # actually in seconds

    def convert(self, counter: X) -> X:
        # convert to unsigned
        counter_diff = (counter - self.sync_counter) & ((1 << 32) - 1)
        return (counter_diff * self.tick_us) + self.sync_time # type: ignore

    def asdict(self) -> dict:
        return asdict(self)

    @staticmethod
    def fromdict(d: dict) -> 'TimeSync':
        return TimeSync(**d)
