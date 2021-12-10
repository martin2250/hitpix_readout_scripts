import dataclasses
import datetime
import json
from dataclasses import dataclass

import h5py
import numpy as np
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


def save_ampout_snr(h5parent: h5py.Group, dset_name: str, config: AmpOutSnrConfig, y_data: np.ndarray, time_offset: float, time_delta: float):
    dset = h5parent.create_dataset(dset_name, data=y_data, compression='gzip')
    dset.attrs['time_offset'] = time_offset
    dset.attrs['time_delta'] = time_delta
    dset.attrs['save_time'] = datetime.datetime.now().isoformat()
    dset.attrs['config'] = json.dumps(config.asdict())


def load_ampout_snr(h5parent: h5py.Group, dset_name: str) -> tuple[AmpOutSnrConfig, np.ndarray, float, float]:
    dset = h5parent[dset_name]
    assert isinstance(dset, h5py.Dataset)
    # load attributes
    config_str = h5parent.attrs['config']
    assert isinstance(config_str, str)
    config = AmpOutSnrConfig.fromdict(json.loads(config_str))
    time_offset = dset.attrs['time_offset']
    time_delta = dset.attrs['time_delta']
    assert isinstance(time_offset, float)
    assert isinstance(time_delta, float)
    y_data = dset[()]
    assert isinstance(y_data, np.ndarray)
    return config, y_data, time_offset, time_delta
