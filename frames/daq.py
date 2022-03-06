import math
from optparse import Option
import queue
import time
from typing import Optional

import numpy as np, numpy._globals
import tqdm
from hitpix import HitPixSetup
from readout import Response
from readout.fast_readout import FastReadout
from readout.instructions import Finish

from hitpix.readout import HitPixReadout
from readout.sm_prog import decode_column_packets, prog_dac_config
from .io import FrameConfig
from .sm_prog import prog_read_frames
import threading

def _decode_responses(
    responses: list[Response],
    frames: list[np.ndarray],
    timestamps: list[np.ndarray],
    timeout: float,
    setup: HitPixSetup,
    reset_counters: bool,
    callback = None,
    progress: Optional[tqdm.tqdm] = None,
    ):
    ctr_max = 1 << setup.chip.bits_counter
    hits_last = None

    for response in responses:
        response.event.wait(timeout)
        assert response.data is not None

        # decode hits
        block_timestamps, block_frames = decode_column_packets(response.data, setup.pixel_columns, setup.chip.bits_adder, setup.chip.bits_counter)
        block_frames = block_frames.reshape(-1, setup.pixel_rows, setup.pixel_columns)

        if not reset_counters:
            if hits_last is None:
                # duplicate first frame -> zero frame at start
                # this makes sure the total number of frames is
                # the same as with reset_counters=True
                hits_last = block_frames[:1]
            hits_last_next = block_frames[-1:]
            block_frames = np.diff(block_frames, axis=0, prepend=hits_last)
            hits_last = hits_last_next

        block_frames = (ctr_max - block_frames) % ctr_max  # counter counts down
        
        if callback:
            callback(block_frames)
        frames.append(block_frames)
        timestamps.append(block_timestamps)

        if progress is not None:
            progress.update()


def read_frames(ro: HitPixReadout, fastreadout: FastReadout, config: FrameConfig, progress: Optional[tqdm.tqdm] = None, callback = None) -> tuple[np.ndarray, np.ndarray]:
    ############################################################################
    # set up readout

    if ro.frequency_mhz_set != config.readout_frequency:
        print(f'setting frequency to {config.readout_frequency=}')
        config.readout_frequency = ro.set_system_clock(config.readout_frequency)
        print(f'actual: {config.readout_frequency}')

    ############################################################################
    # configure readout & chip

    ro.set_threshold_voltage(config.voltage_threshold)
    ro.set_baseline_voltage(config.voltage_baseline)

    ro.sm_exec(prog_dac_config(config.dac_cfg.generate()))

    time.sleep(0.025)

    setup = ro.setup

    ############################################################################
    # prepare statemachine

    pulse_cycles = int(max(1, config.pulse_ns * ro.frequency_mhz / 1000))

    prog_init, prog_readout = prog_read_frames(
        frame_cycles=int(ro.frequency_mhz * config.frame_length_us),
        pulse_cycles=pulse_cycles,
        pause_cycles=int(ro.frequency_mhz * config.pause_length_us),
        reset_counters=config.reset_counters,
        setup=setup,
        frequency=config.readout_frequency,
    )
    prog_readout.append(Finish())

    ro.sm_exec(prog_init)
    ro.sm_write(prog_readout)

    ############################################################################
    # start measurement

    # raw frame duration
    duration_frame = 1e-6 * (config.frame_length_us + config.pause_length_us)
    # time for readout (estimated)
    readout_pixels = setup.pixel_columns * setup.pixel_rows
    duration_frame += 1e-6 * readout_pixels / ro.frequency_mhz
    # X ms per packet for good progress
    frames_per_run = math.ceil(500e-3 / duration_frame)
    if frames_per_run > ((1 << 16) - 1):
        frames_per_run = ((1 << 16) - 1)
    num_runs = math.ceil(config.num_frames / frames_per_run)
    # store real values in config
    config.frames_per_run = frames_per_run
    config.num_frames = frames_per_run * num_runs


    duration_total = num_runs * frames_per_run * duration_frame
    timeout = 15.0 + 3 * duration_total * num_runs

    if progress is not None:
        progress.total = num_runs

    ############################################################################
    # process data

    responses = [fastreadout.expect_response() for _ in range(num_runs)]
    frames = []
    timestamps = []
    
    t_decode = threading.Thread(
        target=_decode_responses,
        daemon=True,
        args=(
            responses,
            frames,
            timestamps,
            timeout,
            setup,
            config.reset_counters,
            callback,
            progress,
        ),
    )
    t_decode.start()

    ############################################################################

    # start measurement
    ro.sm_start(frames_per_run, packets=num_runs)
    ro.wait_sm_idle(timeout)

    t_decode.join(timeout)

    ############################################################################

    frames = np.concatenate(frames)

    timestamps = np.hstack(timestamps)
    # only store timestamp of first row of each frame
    timestamps = timestamps[::setup.pixel_rows]

    times = ro.convert_time(timestamps)

    return frames, times
