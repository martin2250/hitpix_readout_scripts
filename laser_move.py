#!/usr/bin/env python
import os
from typing import Any
import lasersetup.laser
import lasersetup.motion
import threading
import time
import sys

def __get_config_dict_ext() -> dict:
    return {
        'frame_us': 5000.0,
        'pause_us': 0.0,
        'hv': 5.0,
    }
    
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
        '--vdd_driver',
        choices=('default', 'manual'),
        default='default',
        help='use SMU interface to set HV',
    )

    parser.add_argument(
        '--fps',
        type=float, default=5.0,
        help='live view fps',
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
        setattr(a_set, 'completer', ChoicesCompleter(choices_set))
        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args()

    setup_name=args.setup
    args_set=args.set or []
    hv_driver=args.hv_driver
    vssa_driver=args.vssa_driver
    vdd_driver=args.vdd_driver
    fps=args.fps

    import atexit
    import copy
    import sys
    import time
    from pathlib import Path

    import h5py
    import numpy as np
    import tqdm
    import threading

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
    import signal

    ############################################################################

    print_lock = threading.Lock()
    def prints(text):
        '''safe print'''
        with print_lock:
            sys.stdout.write(f'\r\x1b[K{text}\n')
            os.kill(os.getpid(), signal.SIGWINCH)

    ############################################################################

    setup = hitpix.setups[setup_name]

    config_dict: dict[str, Any] = {
        'dac': HitPixDacConfig(**hitpix.defaults.dac_default_hitpix1),
    }
    config_dict.update(**hitpix.defaults.voltages_default)
    config_dict.update(**__get_config_dict_ext())

    util.gridscan.apply_set(config_dict, args_set)

    ############################################################################


    def config_from_dict(config_dict: dict) -> FrameConfig:
        frame_total_us: float = config_dict['frame_us'] + config_dict['pause_us']
        # reconfigure chip every 1s
        num_frames = int(1e6 / frame_total_us)
        # read out to match fps
        frames_per_run = int(1e6 / (fps * frame_total_us))
        if frames_per_run > 200:
            frames_per_run = 200

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
            read_adders=False,
            frames_per_run=frames_per_run
        )

    ############################################################################
    # open readout

    config_readout = util.configuration.load_config()
    serial_port_name, board = config_readout.find_board()

    fastreadout = FastReadout(board.fastreadout_serial_number)
    atexit.register(lambda: prints('closing fastreadout') or fastreadout.close)

    time.sleep(0.05)
    ro = HitPixReadout(serial_port_name, setup)
    ro.initialize()
    atexit.register(lambda: prints('closing readout') or ro.close)

    ############################################################################

    laser = lasersetup.laser.NktPiLaser(board.laser_port)
    motion = lasersetup.motion.load_motion(board.motion_port)

    # configure laser
    laser.trigger_source = laser.TriggerSource.INTERNAL
    laser.trigger_frequency = 50_000
    laser.tune = 0.2
    laser_shutdown = threading.Event()

    def laser_keep():
        state = False
        while not laser_shutdown.is_set():
            s_new = laser.try_enable()
            if s_new != state:
                prints('laser ' + 'on' if s_new else 'off')
            state = s_new
            time.sleep(0.1)
    
    def laser_close():
        laser_shutdown.set()
        laser.state = False
        prints('shut down laser')

    atexit.register(laser_close)
    threading.Thread(target=laser_keep, daemon=True).start()

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

    atexit.register(lambda: prints('shutting down HV') or hv_channel.shutdown)

    ############################################################################

    def set_voltages(config: FrameConfig):
        hv_channel.set_voltage(config.voltage_hv)
        vdd_channel.set_voltage(config.voltage_vdd)
        vssa_channel.set_voltage(config.voltage_vssa)
    
    config = config_from_dict(config_dict)

    # set voltages before starting REPL
    set_voltages(config)

    ############################################################################

    import util.live_view.frames

    live_view = util.live_view.frames.LiveViewFrames(
        sensor_size=(setup.pixel_columns, setup.pixel_rows), # TODO: transposed?
    )

    ############################################################################

    def plot_callback(hits_new):
        live_view.show_frame(
            np.sum(hits_new, axis=0).astype(np.int64),
            config.frames_per_run * config.frame_length_us,
        )

    def target_read():
        while not live_view.closed:
            set_voltages(config)

            frame_total_us = config.frame_length_us + config.pause_length_us
            # reconfigure chip every 1s
            config.num_frames = int(1e6 / frame_total_us)
            # read out to match fps
            frames_per_run = int(1e6 / (fps * frame_total_us))
            if frames_per_run > 200:
                frames_per_run = 200
            config.frames_per_run = frames_per_run

            read_frames(ro, fastreadout, config, callback=plot_callback)
        prints('window closed')

    t_read = threading.Thread(target=target_read, daemon=True)
    t_read.start()

    prints(f'{motion.position=}')
