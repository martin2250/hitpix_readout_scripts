from hitpix1 import HitPix1Readout
from readout.fast_readout import FastReadout
from .io import FrameConfig
from typing import Optional
import tqdm
import numpy as np
import hitpix_roprog
import time
from readout.statemachine import Finish

def read_frames(ro: HitPix1Readout, fastreadout: FastReadout, config: FrameConfig, progress: Optional[tqdm.tqdm] = None) -> tuple[np.ndarray, np.ndarray]:
    ############################################################################
    # configure readout & chip

    ro.set_treshold_voltage(config.voltage_threshold)
    ro.set_baseline_voltage(config.voltage_baseline)

    ro.sm_exec(hitpix_roprog.prog_dac_config(config.dac_cfg.generate(), 7))

    time.sleep(0.025)

    ############################################################################
    # prepare statemachine
    prog_init, prog_readout = hitpix_roprog.prog_read_frames(
        frame_cycles=int(ro.frequency_mhz * config.frame_length_us),
        pulse_cycles=10,
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
    responses = []

    if progress is not None:
        progress.total = num_runs

    # test all voltages
    for _ in range(num_runs):
        # start measurement
        responses.append(fastreadout.expect_response())
        ro.sm_start(config.frames_per_run)
        ro.wait_sm_idle()
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
        block_timestamps, block_frames = hitpix_roprog.decode_column_packets(response.data)
        block_frames = (256 - block_frames) % 256  # counter count down
        frames.append(block_frames.reshape(-1, 24, 24))
        timestamps.append(block_timestamps)

    frames = np.hstack(frames).reshape(-1, 24, 24)
    timestamps = np.hstack(timestamps)

    times = ro.convert_time(timestamps)

    return frames, times
