#!/usr/bin/python
import argparse


def main(
    output_file: str,
    injection_voltage_range: str,
    injections_total: int,
    injections_per_round: int,
    args_scan: list[str],
    args_set: list[str],
):
    import copy
    import time
    from pathlib import Path

    import h5py
    import numpy as np
    import tqdm

    import hitpix.defaults
    import util.configuration
    import util.gridscan
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

    ############################################################################

    def config_from_dict(scan_dict: dict) -> SCurveConfig:
        return SCurveConfig(
            voltage_baseline=scan_dict['baseline'],
            voltage_threshold=scan_dict['threshold'],
            dac_cfg=scan_dict['dac'],
            injection_voltage=injection_voltage,
            injections_per_round=injections_per_round,
            injections_total=injections_total,
        )

    ############################################################################

    path_output = Path(output_file)
    if path_output.exists():
        res = input(
            f'file {path_output} exists, [d]elete , [c]ontinue or [N] abort? (d/c/N): ')
        if res.lower() == 'd':
            path_output.unlink()
        elif res.lower() != 'c':
            exit()

    ############################################################################
    # open readout

    config_readout = util.configuration.load_config()
    serial_port_name, board = config_readout.find_board()

    fastreadout = FastReadout(board.fastreadout_serial_number)
    time.sleep(0.05)
    ro = HitPix1Readout(serial_port_name)
    ro.initialize()

    ############################################################################

    try:
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
                    # perform measurement
                    prog_meas.reset()
                    for _ in range(3):
                        try:
                            res = measure_scurves(
                                ro, fastreadout, config, prog_meas)
                            # store measurement
                            group = file.create_group(group_name)
                            save_scurve(group, config, *res)
                            prog_scan.update()
                            break
                        except Exception as e:
                            prog_scan.write(f'Exception: {repr(e)}')
                            # restart fastreadout on failure
                            fastreadout.close()
                            fastreadout = FastReadout(
                                board.fastreadout_serial_number)
                    else:
                        raise Exception('too many retries')

        else:
            util.gridscan.apply_set(config_dict_template, args_set)
            config = config_from_dict(config_dict_template)

            res = measure_scurves(ro, fastreadout, config, tqdm.tqdm())

            with h5py.File(path_output, 'w') as file:
                group = file.create_group('scurve')
                save_scurve(group, config, *res)
    finally:
        fastreadout.close()

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

    # TODO: implement this
    parser.add_argument(
        '--no-noise',
        action='store_true',
        help='do not record noise hits, inject into whole row',
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
    )
