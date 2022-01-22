import time
from typing import Optional

import numpy as np
import tqdm
from hitpix.readout import HitPixReadout
from readout.fast_readout import FastReadout
from readout.instructions import Finish
from readout.sm_prog import decode_column_packets, prog_dac_config

from .io import SCurveConfig
from .sm_prog import prog_injections_variable


def measure_scurves(ro: HitPixReadout, fastreadout: FastReadout, config: SCurveConfig, progress: Optional[tqdm.tqdm] = None) -> tuple[np.ndarray, Optional[np.ndarray]]:
    ############################################################################
    # calculations
    setup = ro.setup

    assert (setup.pixel_columns % config.simultaneous_injections) == 0
    injection_steps = setup.pixel_columns // config.simultaneous_injections

    ############################################################################
    # configure readout & chip

    # 2.5µs negative pulse with 7.5µs pause
    ro.set_injection_ctrl(
        int(config.injection_pulse_us * ro.frequency_mhz),
        int(config.injection_pause_us * ro.frequency_mhz),
    )

    ro.set_threshold_voltage(config.voltage_threshold)
    ro.set_baseline_voltage(config.voltage_baseline)

    ro.sm_exec(prog_dac_config(config.dac_cfg.generate(), 7))

    time.sleep(0.025)

    ############################################################################
    # prepare statemachine

    prog_injection = prog_injections_variable(
        num_injections=config.injections_per_round,
        shift_clk_div=config.shift_clk_div,
        pulse_cycles=50,
        setup=ro.setup,
        rows=[int(row) for row in config.rows], # np.uint64 to python int
        simultaneous_injections=config.simultaneous_injections,
    )
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
        ro.wait_sm_idle(8.0)

    responses[-1].event.wait(8.0)

    ############################################################################
    # process data

    setup = ro.setup
    ctr_max = 1 << setup.chip.bits_counter

    hits_signal = []
    hits_noise = []

    for response in responses:
        # please pylance type checker
        assert response.data is not None

        # decode hits
        _, hits = decode_column_packets(response.data, setup.pixel_columns, setup.chip.bits_adder, setup.chip.bits_counter)
        hits = (ctr_max - hits) % ctr_max  # counter count down

        # reshape hits
        hits = hits.reshape(
            num_rounds,
            injection_steps,
            len(config.rows),
            setup.pixel_columns,
        )

        # sum over all rounds for this injection voltage
        hits = np.sum(hits, axis=0)

        # extract data from injection steps
        signal = np.zeros((len(config.rows), setup.pixel_columns))
        for injection_step in range(injection_steps):
            # columns are reversed by shift register (0-N == right to left)
            injection_step_rev = injection_steps-1-injection_step
            # extract signal hits
            signal[:,injection_step::injection_steps] = hits[injection_step_rev,:,injection_step::injection_steps]
            # remove signal hits from noise hits
            hits[injection_step_rev,:,injection_step::injection_steps].fill(0)

        # sum up injection steps to get remaining noise hits
        noise = np.sum(hits, axis=0)

        hits_signal.append(signal)
        hits_noise.append(noise)

    hits_signal = np.array(hits_signal)
    hits_noise = np.array(hits_noise)

    return hits_signal, hits_noise
