#!/usr/bin/env python3.10
'''
Inject into pixels of last row (individually).
Record waveforms from ampout pins with oscilloscope.
Check uniformity of pulse heights -> histogram of pulse heights.
Fit gaussian to histogram, position/width = SNR
'''

from typing import Literal


def __get_config_dict_ext() -> dict:
    return {
        # pixel selection
        'column': 0,
        # injection
        'pulse_us': 25.0,
        'pause_us': 50.0,
        'inject_volts': 0.7,
    }


def main(
    output_file: str,
    setup_name: str,
    num_injections: int,
    num_points: int,
    args_scan: list[str],
    args_set: list[str],
    file_exists: Literal['delete', 'continue', 'exit', 'ask'],
    vdd_driver: str,
    vssa_driver: str,
    no_readout: bool,
):
    import atexit
    import copy
    import time
    import sys
    from pathlib import Path

    import h5py
    import numpy as np
    import tqdm

    import ampout_snr.daq
    import hitpix.defaults
    import util.configuration
    import util.gridscan
    import util.lecroy
    import util.voltage_channel
    import util.helpers
    from ampout_snr.io import AmpOutSnrConfig, save_ampout_snr
    from hitpix.readout import HitPixReadout
    import hitpix
    from readout.fast_readout import FastReadout
    from readout.readout import SerialCobsComm

    ############################################################################

    setup = hitpix.setups[setup_name]
    scan_parameters, scan_shape = util.gridscan.parse_scan(args_scan)

    config_dict_template = {
        'dac': setup.chip.dac_config_class.default(),
    }
    config_dict_template.update(**hitpix.defaults.settings_default)
    config_dict_template.update(**__get_config_dict_ext())

    ############################################################################

    def config_from_dict(config_dict: dict) -> AmpOutSnrConfig:
        return AmpOutSnrConfig(
            dac_cfg=config_dict['dac'],
            voltage_baseline=config_dict['baseline'],
            voltage_threshold=config_dict['threshold'],
            voltage_vdd=config_dict['vddd'],
            voltage_vssa=config_dict['vssa'],
            injection_pulse_us=config_dict['pulse_us'],
            injection_pause_us=config_dict['pause_us'],
            injection_volts=config_dict['inject_volts'],
            inject_col=config_dict['column'],
        )

    ############################################################################
    
    path_output = Path(output_file)
    util.helpers.check_output_exists(path_output, file_exists)

    ############################################################################
    # open readout

    config_readout = util.configuration.load_config()
    serial_port_name, board = config_readout.find_board()

    # open fastreadout for 60MHz clock so resets are deasserted
    fastreadout = FastReadout(board.fastreadout_serial_number)
    atexit.register(fastreadout.close)

    scope = util.lecroy.Waverunner8404M(
        util.lecroy.VXI11Connection('192.168.13.178'),
    )

    time.sleep(0.05)
    ro = HitPixReadout(SerialCobsComm(serial_port_name), setup)
    ro.initialize()
    atexit.register(ro.close)

    ro.set_system_clock(25)

    ############################################################################

    if vdd_driver == 'default':
        vdd_driver = board.default_vddd_driver

    if vssa_driver == 'default':
        vssa_driver = board.default_vssa_driver

    vdd_channel = util.voltage_channel.open_voltage_channel(vdd_driver, 'VDD')
    vssa_channel = util.voltage_channel.open_voltage_channel(
        vssa_driver, 'VSSA')

    def set_voltages(config: AmpOutSnrConfig):
        updated = False
        updated |= vdd_channel.set_voltage(config.voltage_vdd)
        updated |= vssa_channel.set_voltage(config.voltage_vssa)
        if updated:
            time.sleep(0.5)


    ############################################################################

    if scan_parameters:
        with h5py.File(path_output, 'a') as file:
            # save scan parameters first
            group_scan = file.require_group('scan')
            util.gridscan.save_scan(group_scan, scan_parameters)
            file.attrs['commandline'] = sys.argv
            # nested progress bars
            prog_scan = tqdm.tqdm(total=np.product(scan_shape), dynamic_ncols=True)
            # scan over all possible combinations
            for idx in np.ndindex(*scan_shape):
                # check if this measurement is already present in file
                dset_name = 'ampout_snr' + ''.join(f'_{i}' for i in idx)
                if dset_name in file:
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
                res = ampout_snr.daq.read_ampout_snr(
                    ro,
                    scope=scope,
                    config=config,
                    num_injections=num_injections,
                    num_points=num_points,
                    no_readout=no_readout,
                )
                # store measurement
                save_ampout_snr(file, dset_name, config, *res)
                prog_scan.update()
    else:
        util.gridscan.apply_set(config_dict_template, args_set)
        config = config_from_dict(config_dict_template)
        set_voltages(config)

        res = ampout_snr.daq.read_ampout_snr(
            ro,
            scope=scope,
            config=config,
            num_injections=num_injections,
            num_points=num_points,
            no_readout=no_readout,
        )

        # store measurement
        with h5py.File(path_output, 'w') as file:
            file.attrs['commandline'] = sys.argv
            save_ampout_snr(file, 'ampout_snr', config, *res)


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

    parser.add_argument(
        '--injections',
        type=int, default=500,
        help='total number of injections to make',
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

    a_set = parser.add_argument(
        '--no_readout',
        action='store_true',
        help='do not read data from scope',
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
        num_injections=args.injections,
        args_scan=args.scan or [],
        args_set=args.set or [],
        file_exists=args.exists,
        vdd_driver=args.vdd_driver,
        vssa_driver=args.vssa_driver,
        num_points=5000,
        no_readout=args.no_readout,
    )
