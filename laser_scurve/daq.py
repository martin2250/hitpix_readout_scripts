import threading
import time
from typing import Optional

import numpy as np
import tqdm
from hitpix import HitPixSetup
from hitpix.readout import HitPixReadout
from readout import Response
from readout import fast_readout
from readout.fast_readout import FastReadout
from readout.instructions import Finish
from readout.sm_prog import decode_column_packets, prog_dac_config

from .io import LaserScurveConfig
from .sm_prog import prog_laser_inject

def _decode_responses(
    responses: list[Response],
    frames: list[np.ndarray],
    timeout: float,
    setup: HitPixSetup,
    ):
    ctr_max = 1 << setup.chip.bits_counter

    for response in responses:
        response.event.wait(timeout)
        assert response.data is not None

        # decode hits
        _, block_frames = decode_column_packets(response.data, setup.pixel_columns, setup.chip.bits_adder, setup.chip.bits_counter)
        block_frames = (ctr_max - block_frames) % ctr_max  # counter counts down
        block_frames = block_frames.reshape(-1, setup.pixel_rows, setup.pixel_columns)
        
        frames.append(block_frames)
    
def measure_laser_scurves(
    ro: HitPixReadout,
    fastreadout: FastReadout,
    config: LaserScurveConfig,
    progress: Optional[tqdm.tqdm] = None
) -> np.ndarray:
    ############################################################################
    # configure readout & chip
    ro.set_injection_ctrl(
        int(config.injection_pulse_us * ro.frequency_mhz),
        int(config.injection_pause_us * ro.frequency_mhz),
    )
    ro.set_baseline_voltage(config.voltage_baseline)
    ro.sm_exec(prog_dac_config(config.dac_cfg.generate(), 7))

    ############################################################################
    # prepare statemachine
    prog_init, prog_readout = prog_laser_inject(
        injections_per_round=config.injections_per_round,
        pulse_cycles=50,
        shift_clk_div=config.shift_clk_div,
        setup=ro.setup,
    )
    prog_readout.append(Finish())

    ro.sm_exec(prog_init)
    ro.sm_write(prog_readout)

    ############################################################################
    # start measurement

    # total number of injection cycles, round up
    num_rounds = int(config.injections_total /
                     config.injections_per_round + 0.99)
    responses = [fastreadout.expect_response() for _ in config.threshold_offsets]

    run_us = config.injections_total * (config.injection_pulse_us + config.injection_pause_us)
    timeout = 2.0 + run_us * 1.5e-6

    if progress is not None:
        progress.total = len(config.threshold_offsets)

    ############################################################################
    # process data

    frames = []
    
    t_decode = threading.Thread(
        target=_decode_responses,
        daemon=True,
        args=(
            responses,
            frames,
            timeout,
            ro.setup,
        ),
    )
    t_decode.start()

    ############################################################################

    # test all voltages
    for threshold_offset in config.threshold_offsets:
        if progress is not None:
            progress.update()
            progress.set_postfix(t=f'{threshold_offset:0.2f}')
        # prepare
        ro.set_threshold_voltage(config.voltage_baseline + threshold_offset)
        time.sleep(config.injection_delay)
        # start measurement
        ro.sm_start(num_rounds)
        ro.wait_sm_idle(timeout)

    t_decode.join()

    assert len(frames) == len(config.threshold_offsets)

    ############################################################################
    
    return np.reshape(
        np.hstack(frames),
        (-1, ro.setup.pixel_rows, ro.setup.pixel_columns),
    )
