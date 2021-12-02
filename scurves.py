#!/usr/bin/python
import argparse
import dataclasses
import datetime
import json
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

################################################################################


def measure_scurves(ro: HitPix1Readout, fastreadout: FastReadout, config: SCurveConfig, progress: Optional[tqdm.tqdm] = None) -> tuple[np.ndarray, np.ndarray]:
    ############################################################################
    # configure readout & chip

    ro.set_injection_ctrl(500, 500)

    ro.set_treshold_voltage(config.voltage_threshold)
    ro.set_baseline_voltage(config.voltage_baseline)

    ro.sm_exec(hitpix_roprog.prog_dac_config(config.dac_cfg.generate(), 7))

    time.sleep(0.1)

    ############################################################################
    # prepare statemachine

    test_injection = hitpix_roprog.TestInjection(
        config.injections_per_round, config.shift_clk_div)
    prog_injection = test_injection.prog_test()
    prog_injection.append(Finish())
    ro.sm_write(prog_injection)

    ############################################################################
    # start measurement

    # total number of injection cycles, round up
    num_rounds = int(config.injections_total /
                     config.injections_per_round + 0.99)
    responses = []

    if progress is not None:
        progress.total = len(config.injection_voltage)

    # test all voltages
    for injection_voltage in config.injection_voltage:
        if progress is not None:
            progress.update()
            progress.set_postfix(v=f'{injection_voltage:0.2f}')
        # prepare
        ro.set_injection_voltage(injection_voltage)
        time.sleep(config.injection_delay)
        # start measurement
        responses.append(fastreadout.expect_response())
        ro.sm_start(num_rounds)
        ro.wait_sm_idle()

    responses[-1].event.wait(5)

    ############################################################################
    # process data

    hits_signal = []
    hits_noise = []

    for response in responses:
        # please pylance type checker
        assert response.data is not None

        # decode hits
        _, hits = hitpix_roprog.decode_column_packets(response.data)
        hits = (256 - hits) % 256  # counter count down

        # sum over all hit frames
        hits = hits.reshape(-1, 48, 24)
        hits = np.sum(hits, axis=0)

        # separate signal and noise columns
        even = hits[:24]
        odd = hits[24:]

        hits_signal.append(np.where(
            np.arange(24) % 2 == 0,
            even, odd,
        ))
        hits_noise.append(np.where(
            np.arange(24) % 2 == 1,
            even, odd,
        ))

    hits_signal = np.array(hits_signal)
    hits_noise = np.array(hits_noise)

    return hits_signal, hits_noise


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


if __name__ == '__main__':
    ############################################################################

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'output_file',
        help='h5 output file',
    )

    parser.add_argument(
        '--h5group',
        default='scurve',
        help='h5 group name',
    )

    def parse_injections(s: str) -> tuple[int, int]:
        if '/' in s:
            total, _, per_round = s.partition('/')
        else:
            total, per_round = s, 50
        return int(total), int(per_round)

    parser.add_argument(
        '--injections', metavar='NUM[/ROUND]',
        default='500/50', type=parse_injections,
        help='total number of injections [/per round]',
    )

    def parse_range(s: str) -> tuple[float, float, int]:
        start, stop, steps = s.split(':')
        start, stop = map(float, (start, stop))
        steps = int(steps)
        assert steps > 0
        return start, stop, steps

    parser.add_argument(
        '--voltages',
        default='0.2:1.6:20', type=parse_range,
        help='range of injection voltages start:stop:count',
    )

    parser.add_argument(
        '--baseline',
        default=1.1, type=float,
        help='baseline voltage (V)',
    )

    parser.add_argument(
        '--threshold',
        default=1.2, type=float,
        help='threshold voltage (V)',
    )

    parser.add_argument(
        '--scan', metavar=('name', 'start:stop:count'),
        action='append',
        nargs=2,
        help='scan parameter',
    )

    parser.add_argument(
        '--no-noise',
        action='store_true',
        help='do not record noise hits, inject into whole row',
    )

    args = parser.parse_args()

    ############################################################################

    injection_voltage = np.linspace(*args.voltages)
    injections_total, injections_per_round = args.injections
    voltage_baseline = args.baseline
    voltage_threshold = args.threshold

    scan_names = []
    scan_values = []
    scan_shape = ()
    if args.scan:
        for name, value_range in args.scan:
            assert name not in scan_names
            scan_names.append(name)
            start, stop, steps=  parse_range(value_range)
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

    ro = HitPix1Readout(
        '/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A6003YJ6-if00-port0')
    ro.initialize()
    fastreadout = FastReadout()

    ############################################################################

    config_common = {
        'injection_voltage':injection_voltage,
        'injections_per_round':injections_per_round,
        'injections_total':injections_total,
    }

    ############################################################################
    
    if args.scan:
        with h5py.File(path_output, 'w') as file:
            group_scan = file.create_group('scan')
            group_scan.attrs['scan_names'] = scan_names
            group_scan.attrs['scan_values'] = [values.tolist() for values in scan_values]
            # nested progress bars
            prog_scan = tqdm.tqdm(total=np.product(scan_shape))
            prog_meas = tqdm.tqdm(leave=None)
            # scan over all possible combinations
            for idx in np.ndindex(scan_shape):
                # check if this measurement is already present in file
                group_name = 'scurve_' + '_'.join(str(i) for i in idx)
                if group_name in file:
                    continue
                # list of modifyable values
                scan_globals = {
                    'dac': HitPix1DacConfig(),
                    'baseline': voltage_baseline,
                    'threshold': voltage_threshold,
                }
                # loop over all scanned values and modify config
                for i_value, (name, values) in enumerate(zip(scan_names, scan_values)):
                    value = values[idx[i_value]]
                    # dac values are all integers
                    if name.startswith('dac.'):
                        value = int(value)
                    exec(f'{name} = {value}', scan_globals)
                # extract all values
                config = SCurveConfig(
                    voltage_baseline=scan_globals['baseline'],
                    voltage_threshold=scan_globals['threshold'],
                    dac_cfg=scan_globals['dac'],
                    **config_common,
                )
                # perform measurement
                prog_meas.reset()
                for ignore in [True, True, False]:
                    try:
                        res = measure_scurves(ro, fastreadout, config, prog_meas)
                        # store measurement
                        group = file.create_group(group_name)
                        save_scurve(group, config, *res)
                        prog_scan.update()
                    except Exception as e:
                        prog_scan.write(f'Exception: {repr(e)}')
                        if ignore:
                            continue
                        raise e
                    break
    
    ############################################################################
   
    else:
        config = SCurveConfig(
            voltage_baseline=voltage_baseline,
            voltage_threshold=voltage_threshold,
            dac_cfg=HitPix1DacConfig(),
            **config_common,
        )

        res = measure_scurves(ro, fastreadout, config, tqdm.tqdm())

        with h5py.File(path_output, 'w') as file:
            group = file.create_group(args.h5group)
            save_scurve(group, config, *res)
