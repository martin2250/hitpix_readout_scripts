import time
from typing import Optional

import numpy as np

from hitpix.readout import HitPixReadout
from hitpix import HitPixColumnConfig
from readout.instructions import Inject
from readout.sm_prog import prog_dac_config, prog_col_config
from .io import AmpOutSnrConfig
from util.lecroy import Waverunner8404M


def read_ampout_snr(
    ro: HitPixReadout,
    scope: Waverunner8404M,
    config: AmpOutSnrConfig,
    num_injections: int,
    num_points: int,
    no_readout: bool = False
) -> tuple[np.ndarray, float, float]:
    setup = ro.setup
    chip = ro.setup.chip

    assert setup.chip_rows == 1

    ro.set_injection_ctrl(
        int(config.injection_pulse_us * ro.frequency_mhz),
        int(config.injection_pause_us * ro.frequency_mhz),
    )
    ro.set_injection_voltage(config.injection_volts)
    ro.set_baseline_voltage(config.voltage_baseline)
    ro.set_threshold_voltage(config.voltage_threshold)

    cc = HitPixColumnConfig(
        inject_row = 1 << (chip.rows - 1),
        inject_col = 1 << config.inject_col,
        ampout_col = 1 << config.inject_col,
        rowaddr    = -1,
    )
    ro.sm_exec(prog_dac_config(config.dac_cfg.generate()))
    ro.sm_exec(prog_col_config(setup.encode_column_config(cc), 2))

    if not no_readout:
        scope.record_sequence(num_injections, num_points, wait=False)

    time.sleep(0.1) # let ampout settle

    ro.sm_exec([
        Inject(num_injections),
    ])
    ro.wait_sm_idle()

    if no_readout:
        return np.zeros((0, 0, 0)), 0, 0

    scope.wait_complete()
    y_data, _, time_offset, time_delta = scope.get_sequence_data([1])

    return y_data[0], time_offset, time_delta
