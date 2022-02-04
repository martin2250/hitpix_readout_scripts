#!/usr/bin/env python
import atexit
import base64
import gzip
import sys
import threading
import time
from typing import Any, Optional

import hitpix
import hitpix.defaults
import numpy as np
import util.configuration
import util.gridscan
import util.helpers
import util.voltage_channel
from hitpix import ReadoutPins
from hitpix.readout import HitPixReadout
from readout import Response
from readout.fast_readout import FastReadout
from readout.instructions import *
from readout.sm_prog import (decode_column_packets, prog_col_config,
                             prog_dac_config, prog_read_matrix)
from rich import print
import rich.progress
from util.sm_multiprog import SmMultiprog

################################################################################

use_hitpix1 = True

voltage_vdd = 1.90
voltage_vssa = 1.25

voltage_injection = 1.4
voltage_baseline = 1.0
voltage_threshold = 1.14

injection_pulse_us = 2.5
injection_pause_us = 7.5

if use_hitpix1:
    dac_cfg = hitpix.HitPix1DacConfig.default()
else:
    dac_cfg = hitpix.HitPix2DacConfig.default()
    dac_cfg.vth = int(255 * voltage_threshold/1.85)

pulse_cycles = 50

# preload all "victim" pixels with N injections
test_injections_prepare = 15

if use_hitpix1:
    setup = hitpix.setups['hitpix1']
else:
    setup = hitpix.setups['hitpix2-1x1']
chip = setup.chip

if use_hitpix1:
    num_runs = 180
    frames_per_run = 256
else:
    num_runs = 20
    frames_per_run = 1024

################################################################################
# open readout

config_readout = util.configuration.load_config()
serial_port_name, board = config_readout.find_board()

fastreadout = FastReadout(board.fastreadout_serial_number)
atexit.register(fastreadout.close)

time.sleep(0.05)
ro = HitPixReadout(serial_port_name, setup)
ro.initialize()
atexit.register(ro.close)


################################################################################

vdd_driver = board.default_vdd_driver
vssa_driver = board.default_vssa_driver

vdd_channel = util.voltage_channel.open_voltage_channel(vdd_driver, 'VDD')
vssa_channel = util.voltage_channel.open_voltage_channel(vssa_driver, 'VSSA')

vdd_channel.set_voltage(voltage_vdd)
vssa_channel.set_voltage(voltage_vssa)

################################################################################

ro.set_injection_ctrl(
    int(injection_pulse_us * ro.frequency_mhz),
    int(injection_pause_us * ro.frequency_mhz),
)

ro.set_baseline_voltage(voltage_baseline)
ro.set_injection_voltage(voltage_injection)
if use_hitpix1:
    ro.set_threshold_voltage(voltage_threshold)

# program DACs
ro.sm_exec(prog_dac_config(dac_cfg.generate(), 7))

################################################################################

# default configuration
cfg = SetCfg(
    shift_rx_invert=True,
    shift_tx_invert=True,
    shift_toggle=True,
    shift_select_dac=False,
    shift_word_len=2 * setup.chip.bits_adder,
    shift_clk_div=0,
    pins=0,
)

def run_test(test_row: int, test_col: int, fout : Optional[Any] = None):
    ################################################################################
    # prepare counters by injecting into all pixels

    prog_prepare = [
        Sleep(50),
        cfg.set_pin(ReadoutPins.ro_rescnt, True),
        Sleep(50),
        cfg,
    ]
    for row_inject in range(setup.pixel_rows):
        if row_inject == test_row:
            continue
        prog_prepare.extend([
            *prog_col_config(setup.encode_column_config(hitpix.HitPixColumnConfig(
                inject_row = 1 << row_inject,
                inject_col = 1 << test_col,
                ampout_col = 0,     # no ampout
                rowaddr = test_row, # disable test row (in case of crosstalk)
            ))),
            Sleep(50),
            cfg.set_pin(ReadoutPins.ro_frame, True),
            Sleep(50),
            Inject(test_injections_prepare),
            Sleep(50),
            cfg,
        ])
    prog_prepare.append(Finish())

    ################################################################################
    # read frame

    prog_readout = prog_read_matrix(
        setup=setup,
        shift_clk_div=0,
        pulse_cycles=50,
        rows=None, # read all rows
    )
    prog_readout.append(Finish())

    ################################################################################
    # inject into test pixel

    prog_inject = [
        *prog_col_config(setup.encode_column_config(hitpix.HitPixColumnConfig(
            inject_row = 1 << test_row,
            inject_col = 1 << test_col,
            ampout_col = 0,
            rowaddr = -1,
        ))),
        Sleep(20),
        cfg.set_pin(ReadoutPins.ro_frame, True),
        Sleep(20),
        Inject(1),
        cfg,
        Sleep(20),
        # no finish, fall through to readout
    ]

    ################################################################################
    # prepare state machine

    multiprog = SmMultiprog(ro)

    offset_prepare = multiprog.add_program(prog_prepare)
    offset_inject  = multiprog.add_program(prog_inject)
    offset_readout = multiprog.add_program(prog_readout)

    ################################################################################

    def decode_response(response: Response) -> np.ndarray:
        assert response.event.wait(10)
        assert response.data

        _, frame = decode_column_packets(
            response.data,
            setup.pixel_columns,
            setup.chip.bits_adder,
            setup.chip.bits_counter,
        )
        frame = frame.reshape(-1, setup.pixel_rows, setup.pixel_columns)
        frame = np.flip(frame, axis=-1)
        return frame

    ################################################################################
    # decode data

    def decode_thread(responses: list[Response]):
        frame_first = decode_response(responses[0])
        i_frame_abs = -1
        for response in responses[1:]:
            frames = decode_response(response)
            frames_prev = np.concatenate((frame_first, frames))
            frame_first = frames[-1:]

            frames_diff = np.diff(frames_prev, axis=0)
            frames_diff = (256 - frames_diff) % 256

            for i_frame, frame_diff in enumerate(frames_diff):
                i_frame_abs += 1
                if (i_frame_abs % 5000) == 0:
                    print(f'{test_row=} {test_col=} {i_frame_abs=}')
                # just a single pixel changed state, this is what we expect
                if np.sum(frame_diff) == 1:
                    continue
                # save frame before and after injection
                frames_save = frames_prev[i_frame:i_frame+2]
                data_save = frames_save.astype(np.uint8).tobytes()
                data_savez = gzip.compress(data_save)
                str_print = f'{test_row} {test_col} {i_frame_abs} '.encode() + base64.b64encode(data_savez) + b'\n'
                if fout:
                    fout.write(str_print)
                print(str_print.decode().strip())
        if fout:
            fout.flush()

    ################################################################################
    # run test

    print(f'running test for {test_row=} {test_col=}')

    responses = [fastreadout.expect_response() for _ in range(num_runs + 1)]

    t_decode = threading.Thread(target=decode_thread, args=(responses,), daemon=True)
    t_decode.start()

    ro.sm_start(1, offset_prepare)
    ro.wait_sm_idle()

    ro.sm_start(1, offset_readout)
    ro.wait_sm_idle()

    for _ in range(num_runs):
        ro.sm_start(frames_per_run, offset_inject)
        ro.wait_sm_idle(10.0)

    t_decode.join()

with open(sys.argv[1], 'wb') as fout:
    workload = np.ndindex(setup.pixel_rows, setup.pixel_columns)
    total = setup.pixel_rows * setup.pixel_columns
    for test_row, test_col in rich.progress.track(workload, total=total):
        run_test(test_row, test_col, fout)
