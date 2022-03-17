#!/usr/bin/env python3
from typing import Any, Literal, Optional, cast


def __get_config_dict_ext() -> dict:
    return {
        'frame_us': 5000.0,
        'pause_us': 0.0,
        'hv': 5.0,
        'reset_counters': 1,
    }


def main(
    output_file: str,
    setup_name: str,
    num_frames: int,
    args_scan: list[str],
    args_set: list[str],
    rows_str: str,
    read_adders: bool,
    file_exists: Literal['delete', 'continue', 'exit', 'ask'],
    sums_only: bool,
    hv_driver: str = 'manual',
    vddd_driver: str = 'manual',
    vdda_driver: str = 'manual',
    vssa_driver: str = 'manual',
    live_fps: Optional[float] = False,
):
    import atexit
    import copy
    import signal
    import sys
    import threading
    import time
    from pathlib import Path

    import h5py
    import numpy as np
    from rich import print
    from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                               TimeElapsedColumn, TimeRemainingColumn)

    import hitpix
    import hitpix.defaults
    import util.configuration
    import util.gridscan
    import util.helpers
    from frames.daq import read_frames
    from frames.io import FrameConfig, save_frame_attrs
    from hitpix.readout import HitPixReadout
    from readout.fast_readout import FastReadout
    from readout.readout import SerialCobsComm
    from util.live_view.frames import LiveViewFrames
    from util.voltage_channel import open_voltage_channel

    ############################################################################

    setup = hitpix.setups[setup_name]

    # rows
    if rows_str == 'all':
        rows = np.array(np.arange(setup.pixel_rows, dtype=np.uint))
    else:
        rows = np.array(util.gridscan.parse_int_range(rows_str), dtype=np.uint)

    # scan
    scan_parameters, scan_shape = util.gridscan.parse_scan(args_scan)

    config_dict_template = {
        'dac': setup.chip.dac_config_class.default(),
    }
    config_dict_template.update(**hitpix.defaults.settings_default)
    config_dict_template.update(**__get_config_dict_ext())

    ############################################################################

    def config_from_dict(config_dict: dict) -> FrameConfig:
        # reading adders only makes sense when resetting counters
        if read_adders:
            assert bool(config_dict['reset_counters'])
        return FrameConfig(
            dac_cfg=config_dict['dac'],
            voltage_baseline=config_dict['baseline'],
            voltage_threshold=config_dict['threshold'],
            voltage_vddd=config_dict['vddd'],
            voltage_vdda=config_dict['vdda'],
            voltage_vssa=config_dict['vssa'],
            voltage_hv=config_dict['hv'],
            num_frames=num_frames,
            frame_length_us=config_dict['frame_us'],
            pause_length_us=config_dict['pause_us'],
            readout_frequency=config_dict['frequency'],
            rows=rows,
            read_adders=read_adders,
            reset_counters=bool(config_dict['reset_counters']),
            pulse_ns=config_dict['pulse_ns'],
            setup_name=setup_name,
        )

    ############################################################################

    path_output = Path(output_file)
    util.helpers.check_output_exists(path_output, file_exists)

    ############################################################################
    # open readout
    print('[yellow]connecting to readout')

    config_readout = util.configuration.load_config()
    serial_port_name, board = config_readout.find_board()

    fastreadout = FastReadout(board.fastreadout_serial_number)
    atexit.register(
        lambda: print('[yellow]closing fastreadout') or fastreadout.close(),
    )

    time.sleep(0.05)
    ro = HitPixReadout(SerialCobsComm(serial_port_name), setup)
    ro.initialize()
    atexit.register(lambda: print('[yellow]closing readout') or ro.close())

    ############################################################################
    print('[yellow]connecting to power supplies')

    if hv_driver == 'default':
        hv_driver = board.default_hv_driver

    if vddd_driver == 'default':
        vddd_driver = board.default_vddd_driver

    if vdda_driver == 'default':
        vdda_driver = board.default_vdda_driver

    if vssa_driver == 'default':
        vssa_driver = board.default_vssa_driver

    hv_channel = open_voltage_channel(hv_driver, 'HV')
    vddd_channel = open_voltage_channel(vddd_driver, 'VDDD')
    vdda_channel = open_voltage_channel(vdda_driver, 'VDDA')
    vssa_channel = open_voltage_channel(vssa_driver, 'VSSA')

    def set_voltages(config: FrameConfig):
        hv_channel.set_voltage(config.voltage_hv)
        vddd_channel.set_voltage(config.voltage_vddd)
        vdda_channel.set_voltage(config.voltage_vdda)
        vssa_channel.set_voltage(config.voltage_vssa)

    atexit.register(
        lambda: print('[yellow]powering off HV') or hv_channel.shutdown(),
    )

    ############################################################################

    evt_stop = threading.Event()

    def handle_int(*_):
        print('interrupted by [red]SIGINT[/red], stopping')
        evt_stop.set()

    signal.signal(signal.SIGINT, handle_int)

    ############################################################################

    callback = None
    if live_fps:
        if read_adders:
            pass
        else:
            view = LiveViewFrames((len(rows), setup.pixel_columns))
            def callback_frames(hits_new: np.ndarray):
                view.show_frame(np.sum(hits_new, axis=0).astype(np.int64), 1.0)
            callback = callback_frames

    ############################################################################

    progress = Progress(
        SpinnerColumn(),
        TextColumn('[progress.description]{task.description}'),
        BarColumn(),
        TextColumn('[progress.percentage]{task.completed:>6.0f}/{task.total:0.0f}'),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        refresh_per_second=5.0,
    )

    file = h5py.File(path_output, 'a')

    with file, progress:
        # save exact command line
        file.attrs['commandline'] = sys.argv
        # save scan parameters, if there are any
        if scan_parameters:
            group_scan = file.require_group('scan')
            util.gridscan.save_scan(group_scan, scan_parameters)
        # progress bars
        task_frames = progress.add_task('Frames', start=False)
        task_scan = progress.add_task(
            'Scan',
            total=np.product(scan_shape),
            visible=bool(scan_parameters),
        )
        # scan over all possible combinations
        for idx in np.ndindex(*scan_shape):
            # stop after ctrl-c
            if evt_stop.is_set():
                break
            # progress bars
            progress.reset(task_frames)
            progress.update(task_scan, advance=1)
            # skip if this measurement is already present in file
            group_name = 'frames' + ''.join(f'_{i}' for i in idx)
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
            assert not sums_only
            # repeat measurement in case of exceptions
            for _ in range(3):
                try:
                    # store measurement
                    group = file.create_group(group_name)
                    save_frame_attrs(group, config)
                    # perform measurement
                    ro.sm_abort()
                    read_frames(
                        ro=ro,
                        fastreadout=fastreadout,
                        config=config,
                        progress=(progress, task_frames),
                        callback=callback,
                        evt_stop=evt_stop,
                        h5group=group
                    )
                    break
                except Exception as e:
                    print(f'[red]exception: {e}, retrying')
                    if group_name in file:
                        del file[group_name]
            else:
                raise Exception('too many retries')
    size_mb = path_output.stat().st_size / (1 << 20)
    print(f'[green]total file size: {size_mb:0.2f} MB')


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
        help='total number of frames to read (-1 == infinite runtime)',
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
        '--vddd_driver',
        choices=('default', 'manual'),
        default='default',
        help='use SMU interface to set HV',
    )

    parser.add_argument(
        '--vdda_driver',
        choices=('default', 'manual'),
        default='default',
        help='use SMU interface to set HV',
    )

    parser.add_argument(
        '--rows',
        default='all',
        help='which rows to read, eg "10:12" or "[10, 11]"',
    )

    def parse_bool(s: str) -> bool:
        if s.strip().lower() in ('1', 'true', 'yes', 'y', 't'):
            return True
        if s.strip().lower() in ('0', 'false', 'no', 'n', 'f'):
            return False
        raise ValueError()

    parser.add_argument(
        '--read_adders',
        type=parse_bool, default=False,
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
        num_frames=args.frames,
        args_scan=args.scan or [],
        args_set=args.set or [],
        rows_str=args.rows,
        read_adders=args.read_adders,
        hv_driver=args.hv_driver,
        vssa_driver=args.vssa_driver,
        vddd_driver=args.vddd_driver,
        vdda_driver=args.vdda_driver,
        file_exists=args.exists,
        sums_only=args.sums_only,
        live_fps=args.live_fps,
    )
