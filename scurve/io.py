import dataclasses
import datetime
import json
from dataclasses import dataclass
from typing import cast

import h5py
import numpy as np
from hitpix1 import *


@dataclass
class SCurveConfig:
    dac_cfg: HitPix1DacConfig
    injection_voltage: np.ndarray
    injections_per_round: int
    injections_total: int
    voltage_baseline: float
    voltage_threshold: float
    shift_clk_div: int = 1
    injection_delay: float = 0.01  # DACs needs around 3ms

    def asdict(self) -> dict:
        d = dataclasses.asdict(self)
        d['dac_cfg'] = dataclasses.asdict(self.dac_cfg)
        d['injection_voltage'] = self.injection_voltage.tolist()
        return d

    @staticmethod
    def fromdict(d: dict) -> 'SCurveConfig':
        dac_cfg = HitPix1DacConfig(**d['dac_cfg'])
        del d['dac_cfg']
        injection_voltage = np.array(d['injection_voltage'])
        del d['injection_voltage']
        return SCurveConfig(
            dac_cfg=dac_cfg,
            injection_voltage=injection_voltage,
            **d,
        )


def save_scurve(h5group: h5py.Group, config: SCurveConfig, hits_signal: np.ndarray, hits_noise: np.ndarray):
    # attributes
    h5group.attrs['save_time'] = datetime.datetime.now().isoformat()
    h5group.attrs['config'] = json.dumps(config.asdict())
    # data
    h5group.create_dataset('hits_signal', data=hits_signal, compression='gzip')
    h5group.create_dataset('hits_noise', data=hits_noise, compression='gzip')


def load_scurve(h5group: h5py.Group) -> tuple[SCurveConfig, np.ndarray, np.ndarray]:
    # load attributes
    config = SCurveConfig.fromdict(
        json.loads(cast(str, h5group.attrs['config'])))
    # load datasets
    dset_signal = h5group['hits_signal']
    dset_noise = h5group['hits_noise']
    assert isinstance(dset_signal, h5py.Dataset)
    assert isinstance(dset_noise, h5py.Dataset)
    hits_signal = dset_signal[()]
    hits_noise = dset_noise[()]
    assert isinstance(hits_signal, np.ndarray)
    assert isinstance(hits_noise, np.ndarray)

    return config, hits_signal, hits_noise
