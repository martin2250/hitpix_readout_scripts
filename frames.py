#!/usr/bin/python
import argparse
import dataclasses
import datetime
import json
from os import EX_NOPERM
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, SupportsIndex, cast, Union, Any
import scipy.optimize

import h5py
import numpy as np
import tqdm
import copy

import hitpix_roprog
from hitpix1 import *
from readout import FastReadout
from statemachine import *

################################################################################


@dataclass
class FrameConfig:
    dac_cfg: HitPix1DacConfig
    voltage_baseline: float
    voltage_threshold: float
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

################################################################################

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

################################################################################


def read_frames(ro: HitPix1Readout, fastreadout: FastReadout, config: FrameConfig, progress: Optional[tqdm.tqdm] = None) -> np.ndarray:
    ############################################################################
    # configure readout & chip

    ro.set_treshold_voltage(config.voltage_threshold)
    ro.set_baseline_voltage(config.voltage_baseline)

    ro.sm_exec(hitpix_roprog.prog_dac_config(config.dac_cfg.generate(), 7))

    time.sleep(0.025)

    ############################################################################
    # prepare statemachine
    prog_init, prog_readout = hitpix_roprog.prog_read_frames(
        frame_cycles=int(ro.frequency_mhz * config.frame_length_us),
        pulse_cycles=10,
        shift_clk_div=config.shift_clk_div,
    )
    prog_readout.append(Finish())

    ro.sm_exec(prog_init)
    ro.sm_write(prog_readout)

    ############################################################################
    # start measurement

    # total number of injection cycles, round up
    num_runs = int(config.num_frames / config.frames_per_run + 0.99)
    responses = []

    if progress is not None:
        progress.total = num_runs

    # test all voltages
    for _ in range(num_runs):
        # start measurement
        responses.append(fastreadout.expect_response())
        ro.sm_start(config.frames_per_run)
        ro.wait_sm_idle()
        if progress is not None:
            progress.update()

    responses[-1].event.wait(5)

    ############################################################################
    # process data

    frames = []

    for response in responses:
        # please pylance type checker
        assert response.data is not None

        # decode hits
        _, hits = hitpix_roprog.decode_column_packets(response.data)
        hits = (256 - hits) % 256  # counter count down
        frames.append(hits.reshape(-1, 24, 24))

    frames = np.hstack(frames).reshape(-1, 24, 24)
    print(frames.shape)
    print(np.sum(frames, axis=0))
    return frames

def __get_config_dict() -> dict:
    return {
        'dac': HitPix1DacConfig(),
        'baseline': 1.1,
        'threshold': 1.2,
        'frame_us': 5000.0,
    }

def __get_help_epilog() -> str:
    help_epilog = 'available parameters for scan/set:\n'
    for name, value in __get_config_dict().items():
        help_epilog += f'{name} = {value}\n' # newline doesn't work, TODO: fix this
    return help_epilog

def __make_config(config_dict: dict, args: argparse.Namespace) -> FrameConfig:
    return FrameConfig(
        dac_cfg=config_dict['dac'],
        voltage_baseline=config_dict['baseline'],
        voltage_threshold=config_dict['threshold'],
        num_frames=args.frames,
        frame_length_us=config_dict['frame_us'],
        read_adders=args.adders,
    )

if __name__ == '__main__':
    ############################################################################

    def parse_range(s: str) -> tuple[float, float, int]:
        start, stop, steps = s.split(':')
        start, stop = map(float, (start, stop))
        steps = int(steps)
        assert steps > 0
        return start, stop, steps

    ############################################################################

    parser = argparse.ArgumentParser(epilog=__get_help_epilog())

    parser.add_argument(
        'output_file',
        help='h5 output file',
    )

    parser.add_argument(
        '--frames',
        type=int, default=5000,
        help='total number of frames to read',
    )

    parser.add_argument(
        '--scan', metavar=('name', 'start:stop:count'),
        action='append',
        nargs=2,
        help='scan parameter',
    )

    parser.add_argument(
        '--set', metavar=('name', 'expression'),
        action='append',
        nargs=2,
        help='set parameter',
    )

    parser.add_argument(
        '--adders',
        action='store_true',
        help='only read adders instead of full matrix',
    )

    args = parser.parse_args()

    ############################################################################

    scan_names = []
    scan_values = []
    scan_shape = ()
    if args.scan:
        for name, value_range in args.scan:
            assert name not in scan_names
            scan_names.append(name)
            start, stop, steps = parse_range(value_range)
            scan_values.append(np.linspace(start, stop, steps))
            scan_shape = scan_shape + (steps,)
    scan_shape = cast(SupportsIndex, scan_shape)

    ############################################################################

    path_output = Path(args.output_file)
    if path_output.exists():
        res = input(f'file {path_output} exists, continue measurement? (y/N): ')
        if res.lower() != 'y':
            exit()

    ############################################################################
    # open readout

    fastreadout = FastReadout()
    time.sleep(0.05)
    ro = HitPix1Readout(
        '/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A904CYCK-if00-port0')
        # '/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A6003YJ6-if00-port0')
    ro.initialize()

    ############################################################################

    if args.scan:
        with h5py.File(path_output, 'w') as file:
            if 'scan' in file:
                group_scan = file['scan']
                assert isinstance(group_scan, h5py.Group)
                scan_names_f = group_scan.attrs['scan_names']
                assert isinstance(scan_names_f, np.ndarray)
                scan_values_f = group_scan.attrs['scan_values']
                assert np.array_equal(scan_names, scan_names_f)
                for name, values in zip(scan_names, scan_values):
                    dset_values = group_scan[name]
                    assert isinstance(dset_values, h5py.Dataset)
                    values_f = dset_values[:]
                    assert isinstance(values_f, np.ndarray)
                    assert np.array_equal(values, values_f)
            else:
                group_scan = file.create_group('scan')
                group_scan.attrs['scan_names'] = scan_names
                for name, values in zip(scan_names, scan_values):
                    group_scan.create_dataset(name, data=values)
            # nested progress bars
            prog_scan = tqdm.tqdm(total=np.product(scan_shape))
            prog_meas = tqdm.tqdm(leave=None)
            # scan over all possible combinations
            for idx in np.ndindex(scan_shape):
                # check if this measurement is already present in file
                group_name = 'frames_' + '_'.join(str(i) for i in idx)
                if group_name in file:
                    continue
                # list of modifyable values
                config_dict = __get_config_dict()
                # loop over all scans and modify config
                for i_value, (name, values) in enumerate(zip(scan_names, scan_values)):
                    value = values[idx[i_value]]
                    # dac values are all integers
                    if name.startswith('dac.'):
                        value = int(value)
                    exec(f'{name} = {value}', config_dict)
                # loop over all sets and modify config
                for name, expression in (args.set or ()):
                    exec(f'{name} = {expression}', config_dict)
                # extract all values
                config = __make_config(config_dict, args)
                # perform measurement
                prog_meas.reset()
                for ignore in [True, True, False]:
                    try:
                        frames = read_frames(ro, fastreadout, config, prog_meas)
                        # store measurement
                        group = file.create_group(group_name)
                        save_frames(group, config, frames)
                        prog_scan.update()
                    except Exception as e:
                        prog_scan.write(f'Exception: {repr(e)}')
                        if ignore:
                            continue
                        raise e
                    break
    
    ############################################################################
   
    else:
        config_dict = __get_config_dict()

        for name, expression in (args.set or ()):
            exec(f'{name} = {expression}', config_dict)

        config = __make_config(config_dict, args)

        frames = read_frames(ro, fastreadout, config, tqdm.tqdm())

        with h5py.File(path_output, 'w') as file:
            group = file.create_group('frames')
            save_frames(group, config, frames)
