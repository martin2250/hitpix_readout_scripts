#!/usr/bin/env python

import sys
from pathlib import Path




if True:  # do not reorder with autopep8 or sortimports
    sys.path.insert(1, str(Path(__file__).parents[1]))

import atexit
import queue
import threading
import time
import bitarray.util
import bitarray

import tqdm
import util.configuration
from readout.sm_prog import prog_shift_dense, prog_sleep
import util.gridscan
import util.voltage_channel
from hitpix.readout import HitPixReadout
import hitpix, hitpix.defaults
from readout import fast_readout
from readout.fast_readout import FastReadout
from readout.instructions import Finish, GetTime, Reset, SetCfg, SetPins, ShiftOut, Sleep

############################################################################

cfg_setup = 'hitpix1'

# cfg_test_string = bitarray.util.urandom(32)
cfg_test_string = bitarray.util.hex2ba('deadbeef', endian='little')
# cfg_test_string = bitarray.util.hex2ba('ff00ff0f', endian='little')
cfg_shift_clk_div = 2
cfg_shift_sample_latency = 0
cfg_rounds = 1
cfg_round_delay = 0.005
cfg_readout_frequency = 50

############################################################################
# open readout

config_readout = util.configuration.load_config()
serial_port_name, board = config_readout.find_board()

fastreadout = FastReadout(board.fastreadout_serial_number)
atexit.register(fastreadout.close)
time.sleep(0.05)

ro = HitPixReadout(serial_port_name, hitpix.setups[cfg_setup])
ro.initialize()
atexit.register(ro.close)

ro.set_system_clock(cfg_readout_frequency)

############################################################################
# readout program

cfg = SetCfg(
    shift_rx_invert = True,
    shift_tx_invert = False,
    shift_toggle = False,
    shift_select_dac = False,
    shift_word_len = 32,
    shift_clk_div = cfg_shift_clk_div,
    shift_sample_latency=cfg_shift_sample_latency,
)
pins = SetPins(0)

num_registers = ro.setup.pixel_columns * ro.setup.chip.bits_adder

if len(cfg_test_string) >= num_registers:
    prog = [
        cfg,
        pins,
        Reset(True, True),
        Sleep(3),
        *prog_shift_dense(cfg_test_string[:num_registers], False),
        *prog_shift_dense(cfg_test_string[num_registers:], True),
        ShiftOut(num_registers, True),
        *prog_sleep(int(10e3 * ro.frequency_mhz)),
    ]
else:
    prog = [
        cfg,
        pins,
        Reset(True, True),
        Sleep(3),
        *prog_shift_dense(cfg_test_string, False),
        ShiftOut(num_registers - len(cfg_test_string), False),
        ShiftOut(len(cfg_test_string), True),
        *prog_sleep(int(10e3 * ro.frequency_mhz)),
    ]

ro.sm_write(prog + [Finish()])

############################################################################

# time.sleep(10)

try:
    while True:
        resp = fastreadout.expect_response()
        
        ro.sm_start(cfg_rounds)
        ro.wait_sm_idle(5.0)

        if not resp.event.wait(5):
            print('no fastreadout response!')
            continue

        assert resp.data

        data_le = bitarray.bitarray(buffer=resp.data)

        print(data_le[::-1])
        print(cfg_test_string)
        # print(resp.data)
        
        time.sleep(cfg_round_delay)
except KeyboardInterrupt:
    pass
except Exception as e:
    print(e)
    time.sleep(1000)