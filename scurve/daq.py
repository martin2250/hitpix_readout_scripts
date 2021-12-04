import time
from typing import Optional

import hitpix_roprog
import numpy as np
import tqdm
from hitpix1 import HitPix1Readout
from readout.fast_readout import FastReadout
from readout.statemachine import Finish

from .io import SCurveConfig


def measure_scurves(ro: HitPix1Readout, fastreadout: FastReadout, config: SCurveConfig, progress: Optional[tqdm.tqdm] = None) -> tuple[np.ndarray, np.ndarray]:
    ############################################################################
    # configure readout & chip

    # 250 ns negative pulse with 4µs pause
    ro.set_injection_ctrl(50, 800)

    ro.set_treshold_voltage(config.voltage_threshold)
    ro.set_baseline_voltage(config.voltage_baseline)

    ro.sm_exec(hitpix_roprog.prog_dac_config(config.dac_cfg.generate(), 7))

    time.sleep(0.1)

    ############################################################################
    # prepare statemachine

    test_injection = hitpix_roprog.TestInjection(
        config.injections_per_round, config.shift_clk_div)
    prog_injection = test_injection.prog_test()
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
        _, hits = hitpix_roprog.decode_column_packets(response.data)
        hits = (256 - hits) % 256  # counter count down

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

    hits_signal = np.array(hits_signal)
    hits_noise = np.array(hits_noise)

    return hits_signal, hits_noise
