#!/usr/bin/python
import argparse


def __get_config_dict_ext() -> dict:
    return {
        'frame_us': 5000.0,
        'pause_us': 0.0,
        'hv': 5.0,
    }


def main(
    output_file: str,
    num_frames: int,
    args_scan: list[str],
    args_set: list[str],
    read_adders: bool,
    hv_driver: str = 'manual'
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
    import util.voltage_channel
    from frames.daq import read_frames
    from frames.io import FrameConfig, save_frames
    from hitpix.hitpix1 import HitPix1DacConfig, HitPix1Readout
    from readout.fast_readout import FastReadout

    ############################################################################

    scan_parameters, scan_shape = util.gridscan.parse_scan(args_scan)

    config_dict_template = {
        'dac': HitPix1DacConfig(**hitpix.defaults.dac_default_hitpix1),
    }
    config_dict_template.update(**hitpix.defaults.voltages_default)
    config_dict_template.update(**__get_config_dict_ext())

    ############################################################################

    def config_from_dict(config_dict: dict) -> FrameConfig:
        return FrameConfig(
            dac_cfg=config_dict['dac'],
            voltage_baseline=config_dict['baseline'],
            voltage_threshold=config_dict['threshold'],
            voltage_hv=config_dict['hv'],
            num_frames=num_frames,
            frame_length_us=config_dict['frame_us'],
            pause_length_us=config_dict['pause_us'],
            read_adders=read_adders,
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

    if hv_driver == 'default':
        hv_driver = board.default_hv_driver

    hv_channel = util.voltage_channel.open_voltage_channel(hv_driver, board)

    ############################################################################

    try:
        if scan_parameters:
            with h5py.File(path_output, 'w') as file:
                # save scan parameters first
                group_scan = file.require_group('scan')
                util.gridscan.save_scan(group_scan, scan_parameters)
                # nested progress bars
                prog_scan = tqdm.tqdm(total=np.product(scan_shape))
                prog_meas = tqdm.tqdm(leave=None)
                # scan over all possible combinations
                for idx in np.ndindex(*scan_shape):
                    # check if this measurement is already present in file
                    group_name = 'frames_' + '_'.join(str(i) for i in idx)
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
                    hv_channel.set_voltage(config.voltage_hv)
                    # perform measurement
                    prog_meas.reset()
                    for _ in range(3):
                        try:
                            frames, times = read_frames(
                                ro, fastreadout, config, prog_meas)
                            # store measurement
                            group = file.create_group(group_name)
                            save_frames(group, config, frames, times)
                            prog_scan.update()
                            break
                        except KeyboardInterrupt:
                            raise KeyboardInterrupt()
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
            hv_channel.set_voltage(config.voltage_hv)

            frames, times = read_frames(ro, fastreadout, config, tqdm.tqdm())

            with h5py.File(path_output, 'w') as file:
                group = file.create_group('frames')
                save_frames(group, config, frames, times)
    except BaseException as e:
        print(e)
    finally:
        fastreadout.close()
        hv_channel.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    a_output_file = parser.add_argument(
        'output_file',
        help='h5 output file',
    )

    parser.add_argument(
        '--frames',
        type=int, default=5000,
        help='total number of frames to read',
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
        '--hv_driver',
        choices=('default', 'manual', 'keithley2400'),
        default='default',
        help='use SMU interface to set HV',
    )

    # TODO: implement this
    parser.add_argument(
        '--adders',
        action='store_true',
        help='only read adders instead of full matrix',
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
        num_frames=args.frames,
        args_scan=args.scan or [],
        args_set=args.set or [],
        read_adders=args.adders,
        hv_driver=args.hv_driver,
    )
