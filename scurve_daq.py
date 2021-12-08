#!/usr/bin/env python3
import argparse
from typing import Literal

def __get_config_dict_ext() -> dict:
    return {
        'pulse_us': 2.5,
        'pause_us': 7.5,
    }

def main(
    output_file: str,
    injection_voltage_range: str,
    injections_total: int,
    injections_per_round: int,
    args_scan: list[str],
    args_set: list[str],
    read_noise: bool,
    file_exists: Literal['delete', 'continue', 'exit', 'ask'],
    vdd_driver: str = 'manual',
    vssa_driver: str = 'manual',
):
    import copy
    import time
    from pathlib import Path

    import h5py
    import numpy as np
    import tqdm
    import atexit

    import hitpix.defaults
    import util.configuration
    import util.gridscan
    import util.voltage_channel
    from hitpix.hitpix1 import HitPix1DacConfig, HitPix1Readout
    from readout.fast_readout import FastReadout
    from scurve.daq import measure_scurves
    from scurve.io import SCurveConfig, save_scurve

    injection_voltage = util.gridscan.parse_range(injection_voltage_range)
    scan_parameters, scan_shape = util.gridscan.parse_scan(args_scan)

    config_dict_template = {
        'dac': HitPix1DacConfig(**hitpix.defaults.dac_default_hitpix1),
    }
    config_dict_template.update(**hitpix.defaults.voltages_default)
    config_dict_template.update(**__get_config_dict_ext())

    ############################################################################

    def config_from_dict(scan_dict: dict) -> SCurveConfig:
        return SCurveConfig(
            voltage_baseline=scan_dict['baseline'],
            voltage_threshold=scan_dict['threshold'],
            dac_cfg=scan_dict['dac'],
            voltage_vdd=scan_dict['vdd'],
            voltage_vssa=scan_dict['vssa'],
            injection_voltage=injection_voltage,
            injections_per_round=injections_per_round,
            injections_total=injections_total,
            injection_pulse_us=scan_dict['pulse_us'],
            injection_pause_us=scan_dict['pause_us'],
        )

    ############################################################################

    path_output = Path(output_file)
    if path_output.exists():
        if file_exists == 'ask':
            res = input(
                f'file {path_output} exists, [d]elete , [c]ontinue or [N] abort? (d/c/N): ')
            if res.lower() == 'd':
                path_output.unlink()
            elif res.lower() != 'c':
                exit()
        elif file_exists == 'delete':
            path_output.unlink()
        elif file_exists != 'continue':
            exit()

    ############################################################################
    # open readout

    config_readout = util.configuration.load_config()
    serial_port_name, board = config_readout.find_board()

    fastreadout = FastReadout(board.fastreadout_serial_number)
    time.sleep(0.05)
    ro = HitPix1Readout(serial_port_name)
    ro.initialize()

    atexit.register(ro.close)
    atexit.register(fastreadout.close)

    ############################################################################

    if vdd_driver == 'default':
        vdd_driver = board.default_vdd_driver

    if vssa_driver == 'default':
        vssa_driver = board.default_vssa_driver

    vdd_channel = util.voltage_channel.open_voltage_channel(vdd_driver, 'VDD')
    vssa_channel = util.voltage_channel.open_voltage_channel(
        vssa_driver, 'VSSA')

    def set_voltages(config: SCurveConfig):
        vdd_channel.set_voltage(config.voltage_vdd)
        vssa_channel.set_voltage(config.voltage_vssa)

    ############################################################################

    if scan_parameters:
        with h5py.File(path_output, 'a') as file:
            # save scan parameters first
            group_scan = file.require_group('scan')
            util.gridscan.save_scan(group_scan, scan_parameters)
            # create nested progress bars
            prog_scan = tqdm.tqdm(total=np.product(scan_shape))
            prog_meas = tqdm.tqdm(leave=None)
            # scan over all possible combinations
            for idx in np.ndindex(*scan_shape):
                # check if this measurement is already present in file
                group_name = 'scurve_' + '_'.join(str(i) for i in idx)
                if group_name in file:
                    # skip
                    prog_scan.update()
                    continue
                # apply scan and set
                config_dict = copy.deepcopy(config_dict_template)
                util.gridscan.apply_scan(config_dict, scan_parameters, idx)
                util.gridscan.apply_set(config_dict, args_set)
                # extract all values
                config = config_from_dict(config_dict)
                set_voltages(config)
                # perform measurement
                prog_meas.reset()
               
                res = measure_scurves(
                    ro, fastreadout, config, read_noise, prog_meas)
                # store measurement
                group = file.create_group(group_name)
                save_scurve(group, config, *res)
                prog_scan.update()

    else:
        util.gridscan.apply_set(config_dict_template, args_set)
        config = config_from_dict(config_dict_template)

        res = measure_scurves(ro, fastreadout, config,
                                read_noise, tqdm.tqdm())

        with h5py.File(path_output, 'w') as file:
            group = file.create_group('scurve')
            save_scurve(group, config, *res)


################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    a_output_file = parser.add_argument(
        'output_file',
        help='h5 output file',
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

    parser.add_argument(
        '--voltages',
        default='0.2:1.6:40',
        help='range of injection voltages start:stop:count[:log]',
    )

    a_scan = parser.add_argument(
        '--scan', metavar=('name=start:stop:count[:log]'),
        action='append',
        help='scan parameter',
    )

    a_set = parser.add_argument(
        '--set', metavar=('name=expression'),
        action='append',
        help='set parameter',
    )

    parser.add_argument(
        '--read_noise',
        action='store_true',
        help='also read noise hits, inject into half row at a time',
    )

    parser.add_argument(
        '--vssa_driver',
        choices=('default', 'manual'),
        default='default',
        help='use SMU interface to set HV',
    )

    parser.add_argument(
        '--vdd_driver',
        choices=('default', 'manual'),
        default='default',
        help='use SMU interface to set HV',
    )

    parser.add_argument(
        '--exists',
        choices=('delete', 'continue', 'exit', 'ask'),
        default='ask',
        help='what to do when file exists',
    )

    try:
        import argcomplete
        from argcomplete.completers import ChoicesCompleter, FilesCompleter

        import hitpix.defaults
        choices_set = []
        for name, value in hitpix.defaults.dac_default_hitpix1.items():
            choices_set.append(f'dac.{name}={value}')
        for name, value in hitpix.defaults.voltages_default.items():
            choices_set.append(f'{name}={value}')
        for name, value in __get_config_dict_ext().items():
            choices_set.append(f'{name}={value}')
        choices_scan = [value + ':' for value in choices_set]
        setattr(a_set, 'completer', ChoicesCompleter(choices_set))
        setattr(a_scan, 'completer', ChoicesCompleter(choices_scan))
        setattr(a_output_file, 'completer', FilesCompleter('h5'))
        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args()

    main(
        output_file=args.output_file,
        injection_voltage_range=args.voltages,
        injections_total=args.injections[0],
        injections_per_round=args.injections[1],
        args_scan=args.scan or [],
        args_set=args.set or [],
        read_noise=args.read_noise,
        file_exists=args.exists,
        vssa_driver=args.vssa_driver,
        vdd_driver=args.vdd_driver,
    )
