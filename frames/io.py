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
class FrameConfig:
    dac_cfg: HitPix1DacConfig
    voltage_baseline: float
    voltage_threshold: float
    voltage_hv: float
    num_frames: int
    frame_length_us: float
    pause_length_us: float
    read_adders: bool
    shift_clk_div: int = 1
    frames_per_run: int = 100

    def asdict(self) -> dict:
        d = dataclasses.asdict(self)
        d['dac_cfg'] = dataclasses.asdict(self.dac_cfg)
        return d

    @staticmethod
    def fromdict(d: dict) -> 'FrameConfig':
        dac_cfg = HitPix1DacConfig(**d['dac_cfg'])
        del d['dac_cfg']
        # TODO: remove this, HV should always be set
        if not 'voltage_hv' in d:
            d['voltage_hv'] = -1
        return FrameConfig(
            dac_cfg=dac_cfg,
            **d,
        )

def save_frames(h5group: h5py.Group, config: FrameConfig, frames: np.ndarray, times: np.ndarray):
    # attributes
    h5group.attrs['save_time'] = datetime.datetime.now().isoformat()
    h5group.attrs['config'] = json.dumps(config.asdict())
    # data
    h5group.create_dataset('frames', data=frames, compression='gzip')
    h5group.create_dataset('times', data=times, compression='gzip')

def load_frames(h5group: h5py.Group) -> tuple[FrameConfig, np.ndarray, np.ndarray]:
    # load attributes
    config_dict = json.loads(cast(str, h5group.attrs['config']))
    # TODO: remove this eventually
    if not 'pause_length_us' in config_dict:
        config_dict['pause_length_us'] = 0.0
    config = FrameConfig.fromdict(config_dict)
    # load frames
    dset_frames = h5group['frames']
    assert isinstance(dset_frames, h5py.Dataset)
    frames = dset_frames[()]
    assert isinstance(frames, np.ndarray)
    # load times
    if 'times' in h5group:
        dset_times = h5group['times']
        assert isinstance(dset_times, h5py.Dataset)
        times = dset_times[()]
        assert isinstance(times, np.ndarray)
    else:
        # TODO: remove this eventually
        times = np.full(frames.shape[0], np.nan)
    return config, frames, times
