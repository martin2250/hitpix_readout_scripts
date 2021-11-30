#!/usr/bin/python
import dataclasses
import datetime
import h5py
import time
from dataclasses import dataclass
import argparse
import json
import pathlib

import numpy as np
import tqdm

import hitpix_roprog
from hitpix1 import *
from readout import FastReadout
from statemachine import *

################################################################################

parser = argparse.ArgumentParser()

parser.add_argument(
    'output_file',
    help='h5 output file',
)

args = parser.parse_args()

################################################################################

path_output = pathlib.Path(args.output_file)
if path_output.exists():
    print(f'file {path_output} exists, override? (y/N)')
    if input().lower() != 'y':
        exit()

################################################################################


@dataclass
class SCurveConfig:
    injection_voltages: np.ndarray
    injections_per_round: int
    injections_total: int
    dac_cfg: HitPix1DacConfig
    voltage_baseline: float
    voltage_threshold: float
    shift_clk_div: int = 1
    injection_delay: float = 0.01  # DACs needs around 3ms


################################################################################

config = SCurveConfig(
    injection_voltages=np.linspace(0.2, 1.6, 50),
    injections_per_round=50,
    injections_total=500,
    voltage_baseline=1.1,
    voltage_threshold=1.2,
    dac_cfg=HitPix1DacConfig(),
)

################################################################################
# open readout

fastreadout = FastReadout()
ro = HitPix1Readout(
    '/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A6003YJ6-if00-port0')
ro.initialize()

################################################################################
# configure readout & chip

ro.set_injection_ctrl(250, 250)

ro.set_treshold_voltage(config.voltage_baseline)
ro.set_baseline_voltage(config.voltage_threshold)

ro.sm_exec(hitpix_roprog.prog_dac_config(config.dac_cfg.generate(), 7))

time.sleep(0.1)

################################################################################
# prepare statemachine

test_injection = hitpix_roprog.TestInjection(
    config.injections_per_round, config.shift_clk_div)
prog_injection = test_injection.prog_test()
prog_injection.append(Finish())
ro.sm_write(prog_injection)

################################################################################
# start measurement

# total number of injection cycles, round up
num_rounds = int(config.injections_total / config.injections_per_round + 0.99)
responses = []

# test all voltages
with tqdm.tqdm(config.injection_voltages) as t:
    t.set_description('testing injection voltages: ')
    for i_voltage, injection_voltage in enumerate(t):
        # prepare
        ro.set_injection_voltage(injection_voltage)
        time.sleep(config.injection_delay)

        # start measurement
        responses.append(fastreadout.expect_response())
        ro.sm_start(num_rounds)

        ro.wait_sm_idle()

responses[-1].event.wait(5)

################################################################################
# process data

hits_signal = []
hits_noise = []

for response in responses:
    # please pylance type checker
    assert response.data is not None

    # decode hits
    timestamps, hits = hitpix_roprog.decode_column_packets(response.data)
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

################################################################################
# save file

with h5py.File(path_output, 'w') as file:
    # dataset group (to add attributes)
    group = file.create_group('injection_ramp')
    # attributes
    group.attrs['measurement_time'] = datetime.datetime.now().isoformat()
    group.attrs['injection_voltages'] = config.injection_voltages
    group.attrs['injections'] = config.injections_total
    group.attrs['injections_per_round'] = config.injections_per_round
    group.attrs['sensor_size'] = [24, 24],
    group.attrs['sendor_dac'] = json.dumps(dataclasses.asdict(config.dac_cfg))
    dset_signal = group.create_dataset('hits_signal', data=hits_signal)
    dset_noise = group.create_dataset('hits_noise', data=hits_noise)
