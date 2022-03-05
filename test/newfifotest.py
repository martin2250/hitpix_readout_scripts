#!/usr/bin/env python

from curses import reset_prog_mode
import sys
from pathlib import Path


if True:  # do not reorder with autopep8 or sortimports
    sys.path.insert(1, str(Path(__file__).parents[1]))

import atexit
import queue
import threading
import time

import tqdm
import util.configuration
import util.gridscan
import util.voltage_channel
from hitpix.readout import HitPixReadout
import hitpix, hitpix.defaults
from readout import fast_readout
from readout.fast_readout import FastReadout
from readout.instructions import Finish, GetTime, Sleep

############################################################################
# open readout

config_readout = util.configuration.load_config()
serial_port_name, board = config_readout.find_board()

fastreadout = FastReadout(board.fastreadout_serial_number)
atexit.register(fastreadout.close)
time.sleep(0.05)

ro = HitPixReadout(serial_port_name, hitpix.setups[hitpix.defaults.setups[0]])
ro.initialize()
atexit.register(ro.close)

############################################################################
# readout program

prog = [
    GetTime(),
    # Sleep(4),
] * 1024
prog.append(Finish())

ro.sm_write(prog)

############################################################################

num_packet = 3

for _ in range(10):
    print('ctrs: ', hex(ro.read_register(0xe0)))

    resp = [fastreadout.expect_response() for _ in range(num_packet)]

    ro.sm_start(1, packets=num_packet)

    for r in resp:
        assert r.event.wait(1.0)
        # print('ctrs: ', hex(ro.read_register(0xe0)))
        assert r.data
        print(len(r.data))

