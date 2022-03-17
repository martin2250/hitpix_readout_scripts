#!/usr/bin/env python3
from typing import Literal


def __get_config_dict_ext() -> dict:
    return {
        'pulse_us': 2.5,
        'pause_us': 17.5,
        'simultaneous_injections': -2,
    }


def main(
    output_file: str,
    setup_name: str,
    injection_voltage_range: str,
    rows_str: str,
    injections_total: int,
    injections_per_round: int,
    args_scan: list[str],
    args_set: list[str],
    file_exists: Literal['delete', 'continue', 'exit', 'ask'],
    vdd_driver: str = 'manual',
    vssa_driver: str = 'manual',
):
    import atexit
    import copy
    import sys
    import time
    from pathlib import Path

    import h5py
    import numpy as np
    import tqdm

    import hitpix
    import hitpix.defaults
    import util.configuration
    import util.gridscan
    import util.helpers
    import util.voltage_channel
    from hitpix.readout import HitPixReadout
    from readout.fast_readout import FastReadout
    from readout.readout import SerialCobsComm
    from scurve.daq import measure_scurves
    from scurve.io import SCurveConfig, save_scurve

    ############################################################################

    setup = hitpix.setups[setup_name]
    # injection voltages
    injection_voltage = util.gridscan.parse_range(injection_voltage_range)
    # rows
    if rows_str == 'all':
        rows = np.arange(setup.pixel_rows, dtype=np.uint)
    else:
        rows = np.array(util.gridscan.parse_int_range(rows_str), dtype=np.uint)
    scan_parameters, scan_shape = util.gridscan.parse_scan(args_scan)

    # check simultaneous_injections parameter
    for scan_parameter in scan_parameters:
        if not scan_parameter.name == 'simultaneous_injections':
            continue
        for value in scan_parameter.values:
            assert (setup.pixel_columns % abs(int(value))) == 0

    config_dict_template = {
        'dac': setup.chip.dac_config_class.default(),
    }
    config_dict_template.update(**hitpix.defaults.settings_default)
    config_dict_template.update(**__get_config_dict_ext())

    ############################################################################

    def config_from_dict(scan_dict: dict) -> SCurveConfig:
        simultaneous_injections = scan_dict['simultaneous_injections']
        assert isinstance(simultaneous_injections, int)
        # check that abs(simultaneous_injections) divides number of columns
        # without remainder
        assert (setup.pixel_columns % abs(simultaneous_injections)) == 0
        # smaller than zero: input == injection steps
        if simultaneous_injections < 0:
            simultaneous_injections = setup.pixel_columns // abs(
                simultaneous_injections)
        # construct config
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
            readout_frequency=scan_dict['frequency'],
            setup_name=setup_name,
            rows=rows,
            pulse_ns=scan_dict['pulse_ns'],
            simultaneous_injections=simultaneous_injections,
        )

    ############################################################################

    path_output = Path(output_file)
    util.helpers.check_output_exists(path_output, file_exists)

    ############################################################################
    # open readout

    config_readout = util.configuration.load_config()
    serial_port_name, board = config_readout.find_board()

    fastreadout = FastReadout(board.fastreadout_serial_number)
    atexit.register(fastreadout.close)

    time.sleep(0.05)
    ro = HitPixReadout(SerialCobsComm(serial_port_name), setup)
    ro.initialize()
    atexit.register(ro.close)

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

    with h5py.File(path_output, 'a') as file:
        file.attrs['commandline'] = sys.argv
        # save scan parameters first
        if scan_parameters:
            group_scan = file.require_group('scan')
            util.gridscan.save_scan(group_scan, scan_parameters)
        # create nested progress bars
        prog_scan = tqdm.tqdm(total=np.product(scan_shape), dynamic_ncols=True)
        prog_meas = tqdm.tqdm(leave=None, dynamic_ncols=True)
        # scan over all possible combinations
        for idx in np.ndindex(*scan_shape):
            prog_scan.update()
            # skip if this measurement is already present in file
            group_name = 'scurve' + ''.join(f'_{i}' for i in idx)
            if group_name in file:
                continue
            # apply scan and set
            config_dict = copy.deepcopy(config_dict_template)
            if scan_parameters:
                util.gridscan.apply_scan(config_dict, scan_parameters, idx)
            util.gridscan.apply_set(config_dict, args_set)
            # extract all values
            config = config_from_dict(config_dict)
            set_voltages(config)
            # perform measurement
            prog_meas.reset()
            res = measure_scurves(ro, fastreadout, config, prog_meas)
            # store measurement
            group = file.create_group(group_name)
            save_scurve(group, config, *res)

################################################################################


if __name__ == '__main__':
    import argparse

    import hitpix.defaults
    parser = argparse.ArgumentParser()

    a_output_file = parser.add_argument(
        'output_file',
        help='h5 output file',
    )

    parser.add_argument(
        '--setup',
        default=hitpix.defaults.setups[0],
        choices=hitpix.defaults.setups,
        help='which hitpix setup to use',
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
        default='0.0:1.2:60',
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
        '--rows',
        default='all',
        help='which rows to read, eg "10:12" or "[10, 11]"',
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
        from argcomplete.completers import FilesCompleter

        def set_completer(prefix, parsed_args, **kwargs):
            choices_set = []
            for name, value in hitpix.defaults.settings_default.items():
                choices_set.append(f'{name}={value}')
            for name, value in __get_config_dict_ext().items():
                choices_set.append(f'{name}={value}')
            for name, value in hitpix.defaults.setup_dac_defaults[parsed_args.setup].items():
                choices_set.append(f'dac.{name}={value}')
            return filter(lambda s: s.startswith(prefix), choices_set)

        def scan_completer(prefix, parsed_args, **kwargs):
            return map(lambda s: s + ':', set_completer(prefix, parsed_args, **kwargs))

        setattr(a_set, 'completer', set_completer)
        setattr(a_scan, 'completer', scan_completer)
        setattr(a_output_file, 'completer', FilesCompleter('h5'))
        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args()

    main(
        output_file=args.output_file,
        setup_name=args.setup,
        injection_voltage_range=args.voltages,
        rows_str=args.rows,
        injections_total=args.injections[0],
        injections_per_round=args.injections[1],
        args_scan=args.scan or [],
        args_set=args.set or [],
        file_exists=args.exists,
        vssa_driver=args.vssa_driver,
        vdd_driver=args.vdd_driver,
    )
