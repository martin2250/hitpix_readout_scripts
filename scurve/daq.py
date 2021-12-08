import time
from typing import Optional

import numpy as np
import tqdm
from hitpix.hitpix1 import HitPix1Readout
from readout.fast_readout import FastReadout
from readout.instructions import Finish
from readout.sm_prog import decode_column_packets, prog_dac_config

from .io import SCurveConfig
from .sm_prog import prog_injections_full, prog_injections_half


def measure_scurves(ro: HitPix1Readout, fastreadout: FastReadout, config: SCurveConfig, read_noise: bool, progress: Optional[tqdm.tqdm] = None) -> tuple[np.ndarray, Optional[np.ndarray]]:
    ############################################################################
    # configure readout & chip

    # 2.5µs negative pulse with 7.5µs pause
    ro.set_injection_ctrl(500, 1500)

    ro.set_treshold_voltage(config.voltage_threshold)
    ro.set_baseline_voltage(config.voltage_baseline)

    ro.sm_exec(prog_dac_config(config.dac_cfg.generate(), 7))

    time.sleep(0.1)

    ############################################################################
    # prepare statemachine

    if read_noise:
        prog_injection = prog_injections_half(config.injections_per_round, config.shift_clk_div)
    else:
        prog_injection = prog_injections_full(config.injections_per_round, config.shift_clk_div)
    prog_injection.append(Finish())
    ro.sm_write(prog_injection)

    ############################################################################
    # start measurement

    # total number of injection cycles, round up
    num_rounds = int(config.injections_total /
                     config.injections_per_round + 0.99)
    responses = []

    if progress is not None:
        progress.total = len(config.injection_voltage)

    # test all voltages
    for injection_voltage in config.injection_voltage:
        if progress is not None:
            progress.update()
            progress.set_postfix(v=f'{injection_voltage:0.2f}')
        # prepare
        ro.set_injection_voltage(injection_voltage)
        time.sleep(config.injection_delay)
        # start measurement
        responses.append(fastreadout.expect_response())
        ro.sm_start(num_rounds)
        ro.wait_sm_idle()

    responses[-1].event.wait(5)

    ############################################################################
    # process data

    hits_signal = []
    hits_noise = []

    for response in responses:
        # please pylance type checker
        assert response.data is not None

        # decode hits
        _, hits = decode_column_packets(response.data)
        hits = (256 - hits) % 256  # counter count down


        if read_noise:
            # sum over all hit frames
            hits = hits.reshape(-1, 48, 24)
            hits = np.sum(hits, axis=0)

            # separate signal and noise columns
            even = hits[:24]
            odd = hits[24:]

            hits_signal.append(np.where(
                np.arange(24) % 2 == 0,
                even, odd,
            ))
            hits_noise.append(np.where(
                np.arange(24) % 2 == 1,
                even, odd,
            ))
        else:
            # sum over all hit frames
            hits = hits.reshape(-1, 24, 24)
            hits = np.sum(hits, axis=0)
            hits_signal.append(hits)

    hits_signal = np.array(hits_signal)

    if read_noise:
        hits_noise = np.array(hits_noise)
        return hits_signal, hits_noise
    else:
        return hits_signal, None
