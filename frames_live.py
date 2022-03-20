#!/usr/bin/env python3

from dataclasses import dataclass
import dataclasses
from typing import Any, Literal, Optional


def __get_config_dict_ext() -> dict:
    return {
        'frame_us': 5000.0,
        'pause_us': 0.0,
        'hv': 5.0,
        'reset_counters': 1,
    }


def main(
    setup_name: str,
    args_set: list[str],
    rows_str: str,
    read_adders: bool,
    fps: float,
    gui: bool,
    hv_driver: str = 'manual',
    vddd_driver: str = 'manual',
    vdda_driver: str = 'manual',
    vssa_driver: str = 'manual',
):
    import atexit
    import signal
    import threading
    import time

    import numpy as np
    from rich import print

    import hitpix
    import hitpix.defaults
    import util.configuration
    import util.gridscan
    import util.helpers
    from frames.daq import read_frames
    from hitpix.readout import HitPixReadout
    from readout.fast_readout import FastReadout
    from readout.readout import SerialCobsComm
    from util.live_view.frames import LiveViewAdders, LiveViewFrames
    from util.voltage_channel import open_voltage_channel

    ############################################################################

    setup = hitpix.setups[setup_name]

    # rows
    if rows_str == 'all':
        if read_adders:
            rows = np.array([], dtype=np.uint)
        else:
            rows = np.array(np.arange(setup.pixel_rows, dtype=np.uint))

    else:
        assert not read_adders
        rows = np.array(util.gridscan.parse_int_range(rows_str), dtype=np.uint)

    config_dict: dict[str, Any] = {
        **hitpix.defaults.settings_default,
        **__get_config_dict_ext(),
        # 'dac': setup.chip.dac_config_class.default(),
    }
    dac = setup.chip.dac_config_class.default()
    util.gridscan.apply_set(config_dict, args_set)

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

    hv_channel.set_voltage(config_dict['hv'])
    vddd_channel.set_voltage(config_dict['vddd'])
    vdda_channel.set_voltage(config_dict['vdda'])
    vssa_channel.set_voltage(config_dict['vssa'])

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

    if read_adders:
        view = LiveViewAdders(setup.pixel_columns, 100)
    else:
        view = LiveViewFrames((len(rows), setup.pixel_columns))

    def callback_frames(hits_new: np.ndarray):
        view.show_frame(np.sum(hits_new, axis=0).astype(np.int64), 1.0)
    callback = callback_frames

    ############################################################################

    if gui:
        from util.live_view.livesliders import SliderValue, LiveSliders
        
        slider_values = []
        for name in ['vssa','vddd','vdda']:
            slider_values.append(SliderValue(
                label=name,
                extent=(1.0, 2.1),
                value=config_dict[name], 
                resolution=0.01,
            ))
        for name in ['threshold', 'baseline']:
            slider_values.append(SliderValue(
                label=name,
                extent=(0.6, 1.4),
                value=config_dict[name],
                resolution=0.01,
            ))
        slider_values.append(SliderValue(
            label='hv',
            extent=(0.0, 120.0),
            value=config_dict['hv'],
            resolution=0.5,
        ))
        slider_values.append(SliderValue(
            label='frame_us',
            extent= (0.0, config_dict['frame_us']),
            value=config_dict['frame_us'],
            resolution=1.0,
        ))
        slider_values.append(SliderValue(
            label='pause_us',
            extent= (0.0, config_dict['pause_us']),
            value=config_dict['pause_us'],
            resolution=1.0,
        ))
        dac_maxvals = {
            'vth': 255,
        }
        for key, value in hitpix.defaults.setup_dac_defaults[setup_name].items():
            if key in ['unlock']:
                continue
            if type(value) is not int:
                continue
            slider_values.append(SliderValue(
                label='dac.' + key,
                extent=(0, dac_maxvals.get(key, 63)),
                value=value,
            ))
        
        def live_callback(s: SliderValue):
            print(s)

        livesliders = LiveSliders(
            slider_values=slider_values,
            callback=live_callback,
        )
        
        
        


    ############################################################################

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


if __name__ == '__main__':
    import argparse

    import hitpix.defaults
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--setup',
        default=hitpix.defaults.setups[0],
        choices=hitpix.defaults.setups,
        help='which hitpix setup to use',
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

    parser.add_argument(
        '--read_adders',
        action='store_true',
        help='only read adders instead of full matrix',
    )

    parser.add_argument(
        '--fps',
        type=float, default=2,
        help='show live image of frames',
    )

    parser.add_argument(
        '--gui',
        action='store_true',
        help='show GUI for settings',
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

        setattr(a_set, 'completer', set_completer)
        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args()

    main(
        setup_name=args.setup,
        args_set=args.set or [],
        rows_str=args.rows,
        read_adders=args.read_adders,
        hv_driver=args.hv_driver,
        vssa_driver=args.vssa_driver,
        vddd_driver=args.vddd_driver,
        vdda_driver=args.vdda_driver,
        fps=args.fps,
        gui=args.gui,
    )
