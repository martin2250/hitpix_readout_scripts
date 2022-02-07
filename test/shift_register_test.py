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
from readout.sm_prog import prog_shift_dense
import util.gridscan
import util.voltage_channel
from hitpix.readout import HitPixReadout
import hitpix, hitpix.defaults
from readout import fast_readout
from readout.fast_readout import FastReadout
from readout.instructions import Finish, GetTime, Reset, SetCfg, Sleep

############################################################################

cfg_setup = 'hitpix1'

cfg_test_string = bitarray.util.urandom(600)
cfg_test_string = bitarray.bitarray('00'*300)
cfg_shift_clk_div = 0
cfg_rounds = 1
cfg_round_delay = 0.5
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

# ro.set_system_clock(cfg_readout_frequency)

############################################################################
# readout program

cfg_int = SetCfg(
    shift_rx_invert = True,
    shift_tx_invert = True,
    shift_toggle = True,
    shift_select_dac = False,
    shift_word_len = 32,
    shift_clk_div = cfg_shift_clk_div,
    pins = 0,
)
prog = [
    cfg_int,
    Reset(True, True),
    Sleep(3),
    *prog_shift_dense(cfg_test_string, False),
]

ro.sm_write(prog)

############################################################################

time.sleep(10)

try:
    while True:
        ro.set_system_clock(100)

        ro.sm_start(cfg_rounds)
        ro.wait_sm_idle(5.0)
        time.sleep(cfg_round_delay)

        ro.set_system_clock(150)

        ro.sm_start(cfg_rounds)
        ro.wait_sm_idle(5.0)
        time.sleep(cfg_round_delay)
except KeyboardInterrupt:
    pass
except:
    time.sleep(1000)