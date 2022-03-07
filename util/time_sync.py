from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class TimeSync:
    sync_counter: int  # readout time register
    sync_time: float  # time.now()
    tick_us: float

    def convert(self, counter: int | np.ndarray) -> float | np.ndarray:
        # convert to unsigned
        counter_diff = (counter - self.sync_counter) & ((1 << 32) - 1)
        return self.sync_time + (counter_diff * self.tick_us) * 1e-6

    def asdict(self) -> dict:
        return asdict(self)

    @staticmethod
    def fromdict(d: dict) -> 'TimeSync':
        return TimeSync(**d)
