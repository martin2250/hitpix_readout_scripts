#!/usr/bin/env python3.10

from typing import Any, Literal, cast
import hitpix.defaults


def __get_config_dict_ext() -> dict:
    config = {
        'hv': 5.0,
        'x': 50.0,
        'y': 8.0,
        'z': 5.0,
        'pulse_us': 2.5,
        'pause_us': 7.5,
    }
    voltages = copy.copy(hitpix.defaults.settings_default)
    del voltages['threshold']
    config.update(voltages)
    return config


def main(
    output_file: str,
    setup_name: str,
    threshold_offsets_str: str,
    injections_total: int,
    injections_per_round: int,
    args_scan: list[str],
    args_set: list[str],
    file_exists: Literal['delete', 'continue', 'exit', 'ask'],
    hv_driver: str = 'manual',
    vdd_driver: str = 'manual',
    vssa_driver: str = 'manual',
):
    import atexit
    import copy
    import math
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
    from laser_scurve.io import LaserScurveConfig, save_laser_scurves
    from laser_scurve.daq import measure_laser_scurves
    from readout.fast_readout import FastReadout
    import lasersetup.laser
    import lasersetup.motion

    ############################################################################

    path_output = Path(output_file)
    util.helpers.check_output_exists(path_output, file_exists)

    ############################################################################

    setup = hitpix.setups[setup_name]
    threshold_offsets = util.gridscan.parse_range(threshold_offsets_str)
    scan_parameters, scan_shape = util.gridscan.parse_scan(args_scan)

    config_dict_template = {
        'dac': setup.chip.dac_config_class.default(),
    }
    config_dict_template.update(**__get_config_dict_ext())

    ############################################################################

    def config_from_dict(scan_dict: dict) -> LaserScurveConfig:
        return LaserScurveConfig(
            dac_cfg=scan_dict['dac'],
            threshold_offsets=threshold_offsets,
            injections_per_round=injections_per_round,
            injections_total=injections_total,
            voltage_baseline=scan_dict['baseline'],
            voltage_hv=scan_dict['hv'],
            voltage_vdd=scan_dict['vdd'],
            voltage_vssa=scan_dict['vssa'],
            injection_pulse_us=scan_dict['pulse_us'],
            injection_pause_us=scan_dict['pause_us'],
            position=cast(Any, tuple(scan_dict[ax] for ax in 'xyz')),
            setup_name=setup_name,
        )

    ############################################################################
    # open readout

    config_readout = util.configuration.load_config()
    serial_port_name, board = config_readout.find_board()

    fastreadout = FastReadout(board.fastreadout_serial_number)
    atexit.register(fastreadout.close)

    time.sleep(0.05)
    ro = HitPixReadout(serial_port_name, setup)
    ro.initialize()
    atexit.register(ro.close)

    ############################################################################

    laser = lasersetup.laser.NktPiLaser(board.laser_port)
    motion = lasersetup.motion.load_motion(board.motion_port)

    # configure laser
    laser.trigger_source = laser.TriggerSource.EXTERNAL_ADJ
    laser.trigger_edge = laser.TriggerEdge.FALLING
    laser.trigger_level = 3.3 / 2
    laser.tune = 0.2

    def laser_shutdown():
        try:
            laser.state = False
        except RuntimeError:
            pass

    atexit.register(laser_shutdown)

    motion.initialize()

    ############################################################################

    if hv_driver == 'default':
        hv_driver = board.default_hv_driver

    if vdd_driver == 'default':
        vdd_driver = board.default_vdd_driver

    if vssa_driver == 'default':
        vssa_driver = board.default_vssa_driver

    hv_channel = util.voltage_channel.open_voltage_channel(hv_driver, 'HV')
    vdd_channel = util.voltage_channel.open_voltage_channel(vdd_driver, 'VDD')
    vssa_channel = util.voltage_channel.open_voltage_channel(
        vssa_driver, 'VSSA')

    def set_voltages(config: LaserScurveConfig):
        hv_channel.set_voltage(config.voltage_hv)
        vdd_channel.set_voltage(config.voltage_vdd)
        vssa_channel.set_voltage(config.voltage_vssa)

    atexit.register(hv_channel.shutdown)

    ############################################################################

    with h5py.File(path_output, 'a') as file:
        # save command line
        file.attrs['commandline'] = sys.argv
        # save scan parameters first
        group_scan = file.require_group('scan')
        util.gridscan.save_scan(group_scan, scan_parameters)
        # nested progress bars
        prog_scan = tqdm.tqdm(total=math.prod(scan_shape), dynamic_ncols=True)
        # scan over all possible combinations
        for idx in np.ndindex(*scan_shape):
            # check if this measurement is already present in file
            group_name = 'laser_scurve' + ''.join(f'_{i}' for i in idx)
            # skip
            if group_name in file:
                prog_scan.update()
                continue

            # apply scan and set
            config_dict = copy.deepcopy(config_dict_template)
            util.gridscan.apply_scan(config_dict, scan_parameters, idx)
            util.gridscan.apply_set(config_dict, args_set)

            # extract all values
            config = config_from_dict(config_dict)

            # prepare measurement
            motion.move_to(*config.position, wait=False)
            prog_scan.set_postfix(pos=','.join(
                f'{p:0.2f}' for p in config.position))
            set_voltages(config)
            ro.sm_abort()
            motion.wait_on_target()

            # repeat measurement when laser is not on after the measurement
            while True:
                if not laser.state:
                    prog_scan.write(
                        'laser not on, waiting for lid to close...')
                    while True:
                        try:
                            laser.state = True
                            prog_scan.write('lid was closed, continuing')
                            break
                        except RuntimeError:
                            time.sleep(0.1)

                # perform measurement
                frames = measure_laser_scurves(
                    ro=ro,
                    fastreadout=fastreadout,
                    config=config,
                )

                if laser.state:
                    break

                prog_scan.write('lid was opened, repeating measurement')

            # store measurement
            group = file.create_group(group_name)
            save_laser_scurves(group, config, frames)
            prog_scan.update()

    ############################################################################


if __name__ == '__main__':
    import argparse

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
        default='0.0:0.4:60',
        help='range of threshold voltages (relative to baseline) start:stop:count[:log]',
    )

    a_set = parser.add_argument(
        '--set', metavar=('name=expression'),
        action='append',
        help='set parameter',
    )

    a_scan = parser.add_argument(
        '--scan', metavar=('name=start:stop:count[:log]'),
        action='append',
        help='scan parameter',
    )

    parser.add_argument(
        '--hv_driver',
        choices=('default', 'manual'),
        default='default',
        help='use SMU interface to set HV',
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
        import copy
        from argcomplete.completers import FilesCompleter

        def set_completer(prefix, parsed_args, **kwargs):
            choices_set = []
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
        threshold_offsets_str=args.voltages,
        injections_total=args.injections[0],
        injections_per_round=args.injections[1],
        args_set=args.set or [],
        args_scan=args.scan or [],
        file_exists=args.exists,
        hv_driver=args.hv_driver,
        vssa_driver=args.vssa_driver,
        vdd_driver=args.vdd_driver,
    )
