import dataclasses
import datetime
import json
import time
from dataclasses import dataclass
from typing import Optional, cast

import h5py
import numpy as np
import tqdm
import hitpix_roprog
from hitpix1 import *
from readout.fast_readout import FastReadout
from readout.statemachine import *

################################################################################


@dataclass
class SCurveConfig:
    dac_cfg: HitPix1DacConfig
    injection_voltage: np.ndarray
    injections_per_round: int
    injections_total: int
    voltage_baseline: float
    voltage_threshold: float
    shift_clk_div: int = 1
    injection_delay: float = 0.01  # DACs needs around 3ms

    def asdict(self) -> dict:
        d = dataclasses.asdict(self)
        d['dac_cfg'] = dataclasses.asdict(self.dac_cfg)
        d['injection_voltage'] = self.injection_voltage.tolist()
        return d

    @staticmethod
    def fromdict(d: dict) -> 'SCurveConfig':
        dac_cfg = HitPix1DacConfig(**d['dac_cfg'])
        del d['dac_cfg']
        injection_voltage = np.array(d['injection_voltage'])
        del d['injection_voltage']
        return SCurveConfig(
            dac_cfg=dac_cfg,
            injection_voltage=injection_voltage,
            **d,
        )

################################################################################


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


def save_scurve(h5group: h5py.Group, config: SCurveConfig, hits_signal: np.ndarray, hits_noise: np.ndarray):
    # attributes
    h5group.attrs['save_time'] = datetime.datetime.now().isoformat()
    h5group.attrs['config'] = json.dumps(config.asdict())
    # data
    h5group.create_dataset('hits_signal', data=hits_signal, compression='gzip')
    h5group.create_dataset('hits_noise', data=hits_noise, compression='gzip')


def load_scurve(h5group: h5py.Group) -> tuple[SCurveConfig, np.ndarray, np.ndarray]:
    # load attributes
    config = SCurveConfig.fromdict(
        json.loads(cast(str, h5group.attrs['config'])))
    # load datasets
    dset_signal = h5group['hits_signal']
    dset_noise = h5group['hits_noise']
    assert isinstance(dset_signal, h5py.Dataset)
    assert isinstance(dset_noise, h5py.Dataset)
    hits_signal = dset_signal[()]
    hits_noise = dset_noise[()]
    assert isinstance(hits_signal, np.ndarray)
    assert isinstance(hits_noise, np.ndarray)

    return config, hits_signal, hits_noise
