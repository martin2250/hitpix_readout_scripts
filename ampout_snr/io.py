import dataclasses
from dataclasses import dataclass
import h5py
import json
import datetime
import numpy as np
from typing import cast

from hitpix.hitpix1 import HitPix1DacConfig

################################################################################

@dataclass
class AmpOutSnrConfig:
    dac_cfg: HitPix1DacConfig
    voltage_baseline: float
    voltage_threshold: float
    voltage_vdd: float
    voltage_vssa: float
    inject_col: int
    injection_pulse_us: float
    injection_pause_us: float
    injection_volts: float

    def asdict(self) -> dict:
        d = dataclasses.asdict(self)
        d['dac_cfg'] = dataclasses.asdict(self.dac_cfg)
        return d

    @staticmethod
    def fromdict(d: dict) -> 'AmpOutSnrConfig':
        dac_cfg = HitPix1DacConfig(**d['dac_cfg'])
        del d['dac_cfg']
        return AmpOutSnrConfig(
            dac_cfg=dac_cfg,
            **d,
        )
