import dataclasses
from dataclasses import dataclass
import h5py
import json
import datetime
import numpy as np
from typing import cast

from hitpix1 import HitPix1DacConfig

################################################################################

@dataclass
class FrameConfig:
    dac_cfg: HitPix1DacConfig
    voltage_baseline: float
    voltage_threshold: float
    voltage_hv: float
    num_frames: int
    frame_length_us: float
    read_adders: bool
    shift_clk_div: int = 1
    frames_per_run: int = 50

    def asdict(self) -> dict:
        d = dataclasses.asdict(self)
        d['dac_cfg'] = dataclasses.asdict(self.dac_cfg)
        return d

    @staticmethod
    def fromdict(d: dict) -> 'FrameConfig':
        dac_cfg = HitPix1DacConfig(**d['dac_cfg'])
        del d['dac_cfg']
        return FrameConfig(
            dac_cfg=dac_cfg,
            **d,
        )

def save_frames(h5group: h5py.Group, config: FrameConfig, frames: np.ndarray):
    # attributes
    h5group.attrs['save_time'] = datetime.datetime.now().isoformat()
    h5group.attrs['config'] = json.dumps(config.asdict())
    # data
    h5group.create_dataset('frames', data=frames, compression='gzip')

def load_frames(h5group: h5py.Group) -> tuple[FrameConfig, np.ndarray]:
    # load attributes
    config = FrameConfig.fromdict(
        json.loads(cast(str, h5group.attrs['config'])))
    # load datasets
    dset_frames = h5group['frames']
    assert isinstance(dset_frames, h5py.Dataset)
    frames = dset_frames[()]
    assert isinstance(frames, np.ndarray)
    return config, frames
