#!/usr/bin/env python3
from typing import Literal, Optional, cast, Any


def __get_config_dict_ext() -> dict:
    return {
        'frame_us': 5000.0,
        'pause_us': 0.0,
        'hv': 5.0,
    }


def main(
    output_file: str,
    setup_name: str,
    num_frames: int,
    args_scan: list[str],
    args_set: list[str],
    read_adders: bool,
    file_exists: Literal['delete', 'continue', 'exit', 'ask'],
    sums_only: bool,
    hv_driver: str = 'manual',
    vdd_driver: str = 'manual',
    vssa_driver: str = 'manual',
    live_fps: Optional[float] = False,
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
    import util.voltage_channel
    from frames.daq import read_frames
    from frames.io import FrameConfig, save_frames
    from hitpix.dac import HitPixDacConfig
    from hitpix.readout import HitPixReadout
    from readout.fast_readout import FastReadout

    ############################################################################

    setup = hitpix.setups[setup_name]
    scan_parameters, scan_shape = util.gridscan.parse_scan(args_scan)

    config_dict_template = {
        'dac': HitPixDacConfig(**hitpix.defaults.dac_default_hitpix1),
    }
    config_dict_template.update(**hitpix.defaults.voltages_default)
    config_dict_template.update(**__get_config_dict_ext())

    ############################################################################

    def config_from_dict(config_dict: dict) -> FrameConfig:
        return FrameConfig(
            dac_cfg=config_dict['dac'],
            voltage_baseline=config_dict['baseline'],
            voltage_threshold=config_dict['threshold'],
            voltage_vdd=config_dict['vdd'],
            voltage_vssa=config_dict['vssa'],
            voltage_hv=config_dict['hv'],
            num_frames=num_frames,
            frame_length_us=config_dict['frame_us'],
            pause_length_us=config_dict['pause_us'],
            read_adders=read_adders,
        )

    ############################################################################

    path_output = Path(output_file)
    if path_output.exists():
        if file_exists == 'ask':
            try:
                res = input(
                    f'file {path_output} exists, [d]elete , [c]ontinue or [N] abort? (d/c/N): ')
            except KeyboardInterrupt:
                exit()
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
    atexit.register(fastreadout.close)

    time.sleep(0.05)
    ro = HitPixReadout(serial_port_name, setup)
    ro.initialize()
    atexit.register(ro.close)

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

    def set_voltages(config: FrameConfig):
        hv_channel.set_voltage(config.voltage_hv)
        vdd_channel.set_voltage(config.voltage_vdd)
        vssa_channel.set_voltage(config.voltage_vssa)

    atexit.register(hv_channel.shutdown)

    ############################################################################

    if scan_parameters:
        with h5py.File(path_output, 'a') as file:
            # save scan parameters first
            group_scan = file.require_group('scan')
            util.gridscan.save_scan(group_scan, scan_parameters)
            file.attrs['commandline'] = sys.argv
            # nested progress bars
            prog_scan = tqdm.tqdm(total=np.product(
                scan_shape), dynamic_ncols=True)
            prog_meas = tqdm.tqdm(leave=None, dynamic_ncols=True)
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
                set_voltages(config)
                # perform measurement
                prog_meas.reset()
                ro.sm_abort()
                frames, times = read_frames(
                    ro, fastreadout, config, prog_meas)
                # sums -> sum up all frames
                if sums_only:
                    shape_new = (1, *frames.shape[1:])
                    frames = frames.sum(axis=0).reshape(*shape_new)
                    times = times[:1]
                # store measurement
                group = file.create_group(group_name)
                save_frames(group, config, frames, times)
                prog_scan.update()
    else:
        util.gridscan.apply_set(config_dict_template, args_set)
        config = config_from_dict(config_dict_template)
        set_voltages(config)

        callback = None
        if live_fps is not None:
            assert live_fps < 30.0 and live_fps > 0.01
            from multiprocessing import Process, Queue
            from queue import Empty

            q_hits = Queue()

            def plot_callback(hits_new):
                q_hits.put(np.sum(hits_new, axis=0).astype(np.int64))
            callback = plot_callback

            frames_per_run = 1e6 / (live_fps * config.frame_length_us)
            config.frames_per_run = max(1, min(250, int(frames_per_run)))
            plot_frame_us = config.frames_per_run * config.frame_length_us
            # matplotlib uses int typing but accepts floats
            plot_pause = cast(int, 0.25/live_fps)

            def format_time(us: float) -> str:
                if us < 1000.0:
                    return f'{us:0.1f} us'
                us /= 1000
                if us < 1000.0:
                    return f'{us:0.1f} ms'
                us /= 1000
                return f'{us:0.1f} s'

            def plot_process():
                hits = np.zeros(
                    (setup.pixel_rows, setup.pixel_columns), np.int64)

                import matplotlib.pyplot as plt
                _, (ax_frame, ax_total) = cast(Any, plt.subplots(1, 2))
                image_frame = ax_frame.imshow(hits)
                image_total = ax_total.imshow(hits)
                plt.colorbar(image_frame, ax=ax_frame)
                plt.colorbar(image_total, ax=ax_total)
                plot_total_us = 0
                ax_frame.set_title(f'hits / {format_time(plot_frame_us)}')

                plt.ion()
                plt.show()

                while True:
                    plt.pause(plot_pause)
                    try:
                        hits_new = q_hits.get_nowait()
                        image_frame.set_data(hits_new)
                        image_frame.set_clim(0, np.max(hits_new))
                        hits += hits_new
                        plot_total_us += plot_frame_us
                        image_total.set_data(hits)
                        image_total.set_clim(0, np.max(hits))
                        ax_total.set_title(f'hits / {format_time(plot_total_us)}')
                    except Empty:
                        pass
            Process(
                target=plot_process,
                daemon=True,
            ).start()

        frames, times = read_frames(
            ro, fastreadout, config, tqdm.tqdm(dynamic_ncols=True), callback)

        if sums_only:
            shape_new = (1, *frames.shape[1:])
            frames = frames.sum(axis=0).reshape(*shape_new)
            times = times[:1]

        with h5py.File(path_output, 'w') as file:
            file.attrs['commandline'] = sys.argv
            group = file.create_group('frames')
            save_frames(group, config, frames, times)


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

    # TODO: implement this
    parser.add_argument(
        '--adders',
        action='store_true',
        help='only read adders instead of full matrix',
    )

    parser.add_argument(
        '--sums_only',
        action='store_true',
        help='sum up all frames instead of saving individual frames',
    )

    parser.add_argument(
        '--exists',
        choices=('delete', 'continue', 'exit', 'ask'),
        default='ask',
        help='what to do when file exists',
    )

    parser.add_argument(
        '--live_fps',
        metavar='FPS',
        type=float, default=None,
        help='show live image of frames',
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

    # live mode is not available during parameter scans
    assert not (args.live_fps and args.scan)

    main(
        output_file=args.output_file,
        setup_name=args.setup,
        num_frames=args.frames,
        args_scan=args.scan or [],
        args_set=args.set or [],
        read_adders=args.adders,
        hv_driver=args.hv_driver,
        vssa_driver=args.vssa_driver,
        vdd_driver=args.vdd_driver,
        file_exists=args.exists,
        sums_only=args.sums_only,
        live_fps=args.live_fps,
    )
