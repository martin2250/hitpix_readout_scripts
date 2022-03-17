import dataclasses
import datetime
import json
from dataclasses import dataclass
from typing import Optional, cast

import h5py
import hitpix
import numpy as np
from util.time_sync import TimeSync

################################################################################


@dataclass
class FrameConfig:
    dac_cfg: hitpix.HitPixDacConfig
    voltage_baseline: float
    voltage_threshold: float
    voltage_vddd: float
    voltage_vdda: float
    voltage_vssa: float
    voltage_hv: float
    num_frames: int
    frame_length_us: float
    pause_length_us: float
    reset_counters: bool
    rows: np.ndarray
    read_adders: bool
    setup_name: str
    readout_frequency: float
    pulse_ns: float
    frames_per_run: int = 1 << 16

    def asdict(self) -> dict:
        d = dataclasses.asdict(self)
        d['dac_cfg'] = dataclasses.asdict(self.dac_cfg)
        d['rows'] = self.rows.tolist()
        return d

    @staticmethod
    def fromdict(d: dict) -> 'FrameConfig':
        if not 'setup_name' in d:
            d['setup_name'] = 'hitpix1'
        setup_name = d['setup_name']
        dac_cfg = hitpix.setups[setup_name].chip.dac_config_class(
            **d['dac_cfg'])
        del d['dac_cfg']
        # TODO: remove this, HV should always be set
        for name in ['voltage_hv', 'voltage_vdd', 'voltage_vssa']:
            if not name in d:
                d[name] = -1.0
        if not 'reset_counters' in d:
            d['reset_counters'] = True
        if 'shift_clk_div' in d:
            del d['shift_clk_div']
        if not 'readout_frequency' in d:
            d['readout_frequency'] = 25.0
        if not 'pulse_ns' in d:
            d['pulse_ns'] = -1.0
        if not 'rows' in d:
            d['rows'] = [-1]
        d['rows'] = np.array(d['rows'], dtype=np.uint)
        if 'voltage_vdd' in d:
            d['voltage_vddd'] = d['voltage_vdda'] = d['voltage_vdd']
            del d['voltage_vdd']
        return FrameConfig(
            dac_cfg=dac_cfg,
            **d,
        )


def save_frame_attrs(
    h5group: h5py.Group,
    config: FrameConfig,
):
    # attributes
    h5group.attrs['save_time'] = datetime.datetime.now().isoformat()
    h5group.attrs['config'] = json.dumps(config.asdict())


def save_frames(
    h5group: h5py.Group,
    config: FrameConfig,
    frames: np.ndarray,
    times: np.ndarray,
    time_sync: TimeSync,
):
    save_frame_attrs(h5group, config)
    # data
    h5group.create_dataset('frames', data=frames, compression='gzip')
    dset_times = h5group.create_dataset(
        'times', data=times, compression='gzip')
    dset_times.attrs['sync'] = json.dumps(time_sync.asdict())


def load_frames(
    h5group: h5py.Group,
    load_times=True,
) -> tuple[FrameConfig, np.ndarray, Optional[np.ndarray], Optional[TimeSync]]:
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
    # skip loading times
    if not load_times:
        return config, frames, None, None
    # load times
    dset_times = h5group['times']
    assert isinstance(dset_times, h5py.Dataset)
    times = dset_times[()]
    assert isinstance(times, np.ndarray)
    # time sync available?
    if not 'sync' in dset_times.attrs:
        return config, frames, times, None
    # load time sync
    sync_dict = cast(str, dset_times.attrs['sync'])
    time_sync = TimeSync.fromdict(json.loads(sync_dict))
    return config, frames, times, time_sync
