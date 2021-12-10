import time
from typing import Optional

import numpy as np

from hitpix.hitpix1 import HitPix1ColumnConfig, HitPix1Readout
from readout.instructions import Inject
from readout.sm_prog import prog_dac_config, prog_col_config
from .io import AmpOutSnrConfig
from util.lecroy import Waverunner8404M


def read_ampout_snr(
    ro: HitPix1Readout,
    scope: Waverunner8404M,
    config: AmpOutSnrConfig,
    num_injections: int,
    num_points: int,
    no_readout: bool = False
) -> tuple[np.ndarray, float, float]:
    ro.set_injection_ctrl(
        int(config.injection_pulse_us * ro.frequency_mhz),
        int(config.injection_pause_us * ro.frequency_mhz),
    )
    ro.set_injection_voltage(config.injection_volts)
    ro.set_baseline_voltage(config.voltage_baseline)
    ro.set_threshold_voltage(config.voltage_threshold)

    cc = HitPix1ColumnConfig(
        inject_row = 1 << 23,
        inject_col = 1 << config.inject_col,
        ampout_col = 1 << config.inject_col,
        rowaddr    = 24,
    )
    ro.sm_exec(prog_dac_config(config.dac_cfg.generate(), 7))
    ro.sm_exec(prog_col_config(cc.generate(), 2))

    scope.record_sequence(num_injections, num_points, wait=False)

    time.sleep(0.1) # let ampout settle

    ro.sm_exec([
        Inject(num_injections),
    ])
    ro.wait_sm_idle()
    scope.wait_complete()

    if no_readout:
        return np.zeros((0, 0, 0)), 0, 0

    y_data, _, time_offset, time_delta = scope.get_sequence_data([1])

    return y_data, time_offset, time_delta
