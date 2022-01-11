import dataclasses
import datetime
import json
from dataclasses import dataclass
from typing import Optional, cast

import h5py
import numpy as np
import hitpix


@dataclass
class LaserScurveConfig:
    dac_cfg: hitpix.HitPixDacConfig
    threshold_offsets: np.ndarray
    injections_per_round: int
    injections_total: int
    voltage_baseline: float
    voltage_hv: float
    voltage_vdd: float
    voltage_vssa: float
    injection_pulse_us: float
    injection_pause_us: float
    position: tuple[float, float, float]
    setup_name: str
    shift_clk_div: int = 0
    injection_delay: float = 0.005  # DACs needs around 3ms

    def asdict(self) -> dict:
        d = dataclasses.asdict(self)
        d['dac_cfg'] = dataclasses.asdict(self.dac_cfg)
        d['threshold_offsets'] = self.threshold_offsets.tolist()
        return d

    @staticmethod
    def fromdict(d: dict) -> 'LaserScurveConfig':
        if not 'setup_name' in d:
            d['setup_name'] = 'hitpix1'
        setup_name = d['setup']
        dac_cfg = hitpix.setups[setup_name].chip.dac_config_class(**d['dac_cfg'])
        del d['dac_cfg']
        threshold_offsets = np.array(d['threshold_offsets'])
        del d['threshold_offsets']
        position = cast(tuple[float, float, float], tuple(d['position']))
        del d['position']
        return LaserScurveConfig(
            dac_cfg=dac_cfg,
            threshold_offsets=threshold_offsets,
            position=position,
            **d,
        )


def save_laser_scurves(h5group: h5py.Group, config: LaserScurveConfig, frames: np.ndarray):
    # attributes
    h5group.attrs['save_time'] = datetime.datetime.now().isoformat()
    h5group.attrs['config'] = json.dumps(config.asdict())
    # data
    h5group.create_dataset('frames', data=frames, compression='gzip')

def load_laser_scurves(h5group: h5py.Group) -> tuple[LaserScurveConfig, np.ndarray]:
    # load attributes
    config_dict = json.loads(cast(str, h5group.attrs['config']))
    config = LaserScurveConfig.fromdict(config_dict)
    # load frames
    dset_frames = h5group['frames']
    assert isinstance(dset_frames, h5py.Dataset)
    frames = dset_frames[()]
    assert isinstance(frames, np.ndarray)
    return config, frames

