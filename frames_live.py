#!/usr/bin/env python3

from dataclasses import dataclass, field
import dataclasses
import math
import queue
from threading import Thread
from typing import Any, Callable, Literal, Optional

from matplotlib.pyplot import pause
from numpy import isin
from frames.daq_live import FramesLiveSmConfig, live_decode_responses, live_write_statemachine
from hitpix import HitPixColumnConfig

from readout.sm_prog import prog_col_config, prog_dac_config


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
    test_ampout: bool,
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
            rows = []
        else:
            rows = list(range(setup.pixel_rows))

    else:
        assert not read_adders
        rows = util.gridscan.parse_int_range(rows_str)

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

    @atexit.register
    def close_fastreadout():
        print('[yellow]closing fastreadout')
        fastreadout.close()

    time.sleep(0.05)
    ro = HitPixReadout(SerialCobsComm(serial_port_name), setup)
    ro.initialize()

    @atexit.register
    def exit_readout():
        print('[yellow]closing readout')
        ro.sm_abort()
        ro.close()

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

    voltage_channels = {
        'hv':   (hv_channel, 0.0, 200.0),
        'vddd': (vddd_channel, 0.0, 2.05),
        'vdda': (vdda_channel, 0.0, 2.05),
        'vssa': (vssa_channel, 0.0, 2.05),
    }

    hv_channel.set_voltage(config_dict['hv'])
    vddd_channel.set_voltage(config_dict['vddd'])
    vdda_channel.set_voltage(config_dict['vdda'])
    vssa_channel.set_voltage(config_dict['vssa'])

    atexit.register(
        lambda: print('[yellow]powering off HV') or hv_channel.shutdown(),
    )

    ############################################################################

    # evt_stop = threading.Event()

    # def handle_int(*_):
    #     print('interrupted by [red]SIGINT[/red], stopping')
    #     evt_stop.set()

    # signal.signal(signal.SIGINT, handle_int)

    ############################################################################

    view = None
    if not test_ampout:
        if read_adders:
            view = LiveViewAdders(setup.pixel_columns, 100)
        else:
            view = LiveViewFrames((len(rows), setup.pixel_columns))
    
    frame_us_per_packet = -1.0

    def callback_frames(hits_new: np.ndarray):
        if view is None:
            return
        view.show_frame(hits_new.astype(np.int64), frame_us_per_packet)

    ############################################################################

    command_queue = queue.Queue()

    ############################################################################

    if gui:
        from util.live_view.livesliders import SliderValue, LiveSliders

        slider_values = []
        for name in ['vssa', 'vddd', 'vdda']:
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
        if not test_ampout:
            slider_values.append(SliderValue(
                label='frame_us',
                extent=(0.0, config_dict['frame_us']),
                value=config_dict['frame_us'],
                resolution=1.0,
            ))
            slider_values.append(SliderValue(
                label='pause_us',
                extent=(0.0, config_dict['pause_us']),
                value=config_dict['pause_us'],
                resolution=1.0,
            ))
            slider_values.append(SliderValue(
                label='pulse_ns',
                extent=(0.0, 1500.0),
                value=config_dict['pulse_ns'],
                resolution=25.0,
            ))
        if test_ampout:
            slider_values.append(SliderValue(
                label='ampout_col',
                extent=(0, setup.pixel_columns - 1),
                value=0,
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
            command_queue.put((s.label, s.value))

        def run_sliders():
            import tkinter
            livesliders = LiveSliders(
                slider_values=slider_values,
                callback=live_callback,
            )
            tkinter.mainloop()

        thread_sliders = Thread(target=run_sliders, daemon=True)
        thread_sliders.start()

    ############################################################################
    # configure readout & chip

    freq = config_dict['frequency']
    print(f'[yellow]setting system frequency to {freq:0.2f} MHz')
    ro.set_system_clock(freq)

    ro.set_threshold_voltage(config_dict['threshold'])
    ro.set_baseline_voltage(config_dict['baseline'])

    ro.sm_exec(prog_dac_config(dac.generate()))

    ############################################################################
    # prepare statemachine

    sm_config = FramesLiveSmConfig(
        pulse_ns=config_dict['pulse_ns'],
        frame_us=config_dict['frame_us'],
        pause_us=config_dict['pause_us'],
    )

    reset_counters = bool(config_dict['reset_counters'])
    time_frame_us = live_write_statemachine(
        ro,
        sm_config,
        rows,
        reset_counters,
    )

    ############################################################################
    # prepare response parser

    if not test_ampout:
        response_queue = queue.Queue()
        fastreadout.orphan_response_queue = response_queue

        thread_decode = threading.Thread(
            target=live_decode_responses,
            daemon=True,
            kwargs={
                'response_queue': response_queue,
                'setup': ro.setup,
                'reset_counters': reset_counters,
                'num_rows': len(rows),
                'callback': callback_frames,
            },
        )
        thread_decode.start()

    ############################################################################
    # starting statemachine with correct number of runs

    ampout_col = 0

    def start_sm():
        nonlocal frame_us_per_packet
        if test_ampout:
            colcfg = HitPixColumnConfig(ampout_col=(1 << ampout_col))
            data = setup.encode_column_config(colcfg)
            ro.sm_exec(prog_col_config(data))
        else:
            num_runs = int(1e6 / (fps * time_frame_us))
            frame_us_per_packet = num_runs * sm_config.frame_us
            ro.sm_start(runs=num_runs, packets=0)
    
    start_sm()

    ############################################################################

    @dataclass
    class CommandBuffer:
        matches: Callable[[str], bool]
        callback: Callable = lambda _: None
        values: dict[str, float] = field(default_factory=lambda: {})
        time_exec: Optional[float] = None

    def buffer_cb_ro_voltage(values: dict[str, float]):
        for name, value in values.items():
            if not (isinstance(name, str) and isinstance(value, float)):
                continue
            if name == 'threshold' and (0.0 <= value <= 1.8):
                config_dict[name] = value
                print(f'{name} = {value}')
                ro.set_threshold_voltage(value)
            if name == 'baseline' and (0.0 <= value <= 1.8):
                config_dict[name] = value
                print(f'{name} = {value}')
                ro.set_baseline_voltage(value)

    buffer_ro_voltage = CommandBuffer(
        matches=lambda name: name in {'threshold', 'baseline'},
        callback=buffer_cb_ro_voltage,
    )

    def buffer_cb_power_supply(values: dict[str, float]):
        for name, value in values.items():
            if not (isinstance(name, str) and isinstance(value, float)):
                continue
            if name not in voltage_channels:
                continue
            channel, vmin, vmax = voltage_channels[name]
            if not (vmin <= value <= vmax):
                continue
            config_dict[name] = value
            print(f'{name} = {value}')
            channel.set_voltage(value)

    buffer_power_supply = CommandBuffer(
        matches=lambda name: name in {'vddd', 'vdda', 'vssa', 'hv'},
        callback=buffer_cb_power_supply,
    )

    def buffer_cb_dac(values: dict[str, float]):
        update = False
        for name, value in values.items():
            if not (isinstance(name, str) and isinstance(value, int)):
                continue
            name = name.removeprefix('dac.')
            if not hasattr(dac, name):
                continue
            setattr(dac, name, value)
            print(f'{name} = {value}')
            update = True
        if not update:
            return
        ro.sm_soft_abort()
        ro.wait_sm_idle()
        ro.sm_exec(prog_dac_config(dac.generate()))
        start_sm()

    buffer_dac = CommandBuffer(
        matches=lambda name: name.startswith('dac.'),
        callback=buffer_cb_dac,
    )

    def buffer_cb_sm_prog(values: dict[str, float | int]):
        nonlocal time_frame_us, ampout_col
        update = False
        for name, value in values.items():
            if name == 'ampout_col' and isinstance(value, int):
                ampout_col = value
                print(f'{name} = {value}')
                continue
            if not (isinstance(name, str) and isinstance(value, float)):
                continue
            if not hasattr(sm_config, name):
                continue
            setattr(sm_config, name, value)
            print(f'{name} = {value}')
            update = True
        if not update:
            return
        ro.sm_soft_abort()
        ro.wait_sm_idle()
        if not test_ampout:
            time_frame_us = live_write_statemachine(
                ro,
                sm_config,
                rows,
                reset_counters,
            )
        start_sm()
            

    buffer_sm_prog = CommandBuffer(
        matches=lambda name: name in {'pulse_ns', 'frame_us', 'pause_us', 'ampout_col'},
        callback=buffer_cb_sm_prog,
    )

    buffers = [buffer_ro_voltage, buffer_power_supply, buffer_dac, buffer_sm_prog]

    def process_command(command):
        if not isinstance(command, tuple):
            return
        name, value = command
        for buffer in buffers:
            if not buffer.matches(name):
                continue
            buffer.values[name] = value
            if buffer.time_exec is None:
                buffer.time_exec = time.monotonic() + 2. / fps

    
    while True:
        timeout = None
        now = time.monotonic()
        for buffer in buffers:
            if buffer.time_exec is None:
                continue
            tdiff = now - buffer.time_exec
            if tdiff < 0:
                buffer.callback(buffer.values)
                buffer.time_exec = None
                buffer.values.clear()
            elif timeout is None or tdiff < timeout:
                timeout = tdiff
        try:
            command = command_queue.get(block=True, timeout=timeout)
            process_command(command)
            while True:
                command = command_queue.get_nowait()
                process_command(command)
        except queue.Empty:
            pass


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
        '--test_ampout',
        action='store_true',
        help='test ampout instead of reading frames',
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
        test_ampout=args.test_ampout,
        hv_driver=args.hv_driver,
        vssa_driver=args.vssa_driver,
        vddd_driver=args.vddd_driver,
        vdda_driver=args.vdda_driver,
        fps=args.fps,
        gui=args.gui,
    )
