import time
from typing import Optional

import numpy as np
import tqdm
from readout.fast_readout import FastReadout
from readout.instructions import Finish

from hitpix.hitpix1 import HitPix1Readout
from readout.sm_prog import decode_column_packets, prog_dac_config
from .io import FrameConfig
from .sm_prog import prog_read_frames


def read_frames(ro: HitPix1Readout, fastreadout: FastReadout, config: FrameConfig, progress: Optional[tqdm.tqdm] = None) -> tuple[np.ndarray, np.ndarray]:
    ############################################################################
    # configure readout & chip

    ro.set_threshold_voltage(config.voltage_threshold)
    ro.set_baseline_voltage(config.voltage_baseline)

    ro.sm_exec(prog_dac_config(config.dac_cfg.generate(), 7))

    time.sleep(0.025)

    ############################################################################
    # prepare statemachine
    prog_init, prog_readout = prog_read_frames(
        frame_cycles=int(ro.frequency_mhz * config.frame_length_us),
        pulse_cycles=50,
        shift_clk_div=config.shift_clk_div,
        pause_cycles=int(ro.frequency_mhz * config.pause_length_us),
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

    if progress is not None:
        progress.total = num_runs

    # test all voltages
    for _ in range(num_runs):
        # start measurement
        ro.sm_start(config.frames_per_run)
        ro.wait_sm_idle(1.0 + duration_run)
        if progress is not None:
            progress.update()

    responses[-1].event.wait(5)

    ############################################################################
    # process data

    frames = []
    timestamps = []

    for response in responses:
        # please pylance type checker
        assert response.data is not None

        # decode hits
        block_timestamps, block_frames = decode_column_packets(response.data)
        block_frames = (256 - block_frames) % 256  # counter count down
        frames.append(block_frames.reshape(-1, 24, 24))
        timestamps.append(block_timestamps)

    frames = np.hstack(frames).reshape(-1, 24, 24)
    timestamps = np.hstack(timestamps)
    # only store timestamp of first row
    timestamps = timestamps[::24]

    times = ro.convert_time(timestamps)

    return frames, times
