import time
from typing import Optional

import numpy as np
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
    callback = None,
    ):
    ctr_max = 1 << setup.chip.bits_counter
    
    for response in responses:
        response.event.wait(timeout)
        assert response.data is not None

        # decode hits
        block_timestamps, block_frames = decode_column_packets(response.data, setup.pixel_columns, setup.chip.bits_adder, setup.chip.bits_counter)
        block_frames = (ctr_max - block_frames) % ctr_max  # counter counts down
        block_frames = block_frames.reshape(-1, setup.pixel_rows, setup.pixel_columns)
        
        if callback:
            callback(block_frames)
        frames.append(block_frames)
        timestamps.append(block_timestamps)


def read_frames(ro: HitPixReadout, fastreadout: FastReadout, config: FrameConfig, progress: Optional[tqdm.tqdm] = None, callback = None) -> tuple[np.ndarray, np.ndarray]:
    ############################################################################
    # configure readout & chip

    ro.set_threshold_voltage(config.voltage_threshold)
    ro.set_baseline_voltage(config.voltage_baseline)

    ro.sm_exec(prog_dac_config(config.dac_cfg.generate(), 7))

    time.sleep(0.025)

    setup = ro.setup

    ############################################################################
    # prepare statemachine
    prog_init, prog_readout = prog_read_frames(
        frame_cycles=int(ro.frequency_mhz * config.frame_length_us),
        pulse_cycles=50,
        shift_clk_div=config.shift_clk_div,
        pause_cycles=int(ro.frequency_mhz * config.pause_length_us),
        setup=setup,
    )
    prog_readout.append(Finish())

    ro.sm_exec(prog_init)
    ro.sm_write(prog_readout)

    ############################################################################
    # start measurement

    # total number of injection cycles, round up
    num_runs = int(config.num_frames / config.frames_per_run + 0.99)
    responses = [fastreadout.expect_response() for _ in range(num_runs)]

    duration_run = 1e-6 * config.frames_per_run * (config.frame_length_us + config.pause_length_us)
    timeout = 5.0 + duration_run

    if progress is not None:
        progress.total = num_runs

    ############################################################################
    # process data

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
            callback,
        ),
    )
    t_decode.start()

    ############################################################################
    # test all voltages

    for _ in range(num_runs):
        # start measurement
        ro.sm_start(config.frames_per_run)
        ro.wait_sm_idle(timeout)
        if progress is not None:
            progress.update()

    t_decode.join(timeout)

    ############################################################################

    frames = np.hstack(frames).reshape(-1, setup.pixel_rows, setup.pixel_columns)
    timestamps = np.hstack(timestamps)
    # only store timestamp of first row of each frame
    timestamps = timestamps[::setup.pixel_rows]

    times = ro.convert_time(timestamps)

    return frames, times
