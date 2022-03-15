#!/usr/bin/env python

import sys
from pathlib import Path



if True:  # do not reorder with autopep8 or sortimports
    sys.path.insert(1, str(Path(__file__).parents[1]))

import atexit
import queue
import threading
import time

from readout.sm_prog import prog_sleep
import tqdm
import util.configuration
import util.gridscan
import util.voltage_channel
from hitpix.readout import HitPixReadout
import hitpix, hitpix.defaults
from readout.readout import SerialCobsComm
from readout.fast_readout import FastReadout
from readout.instructions import Finish, GetTime, Inject, ShiftIn16, Sleep, count_cycles
from rich import print

############################################################################
# open readout

config_readout = util.configuration.load_config()
serial_port_name, board = config_readout.find_board()

fastreadout = FastReadout(board.fastreadout_serial_number)
atexit.register(fastreadout.close)
time.sleep(0.05)

ro = HitPixReadout(SerialCobsComm(serial_port_name), hitpix.setups[hitpix.defaults.setups[0]])
ro.debug_responses = True
ro.initialize()
atexit.register(ro.close)


############################################################################

# f = 1e6*ro.set_system_clock(175.0)
f = 175.0

print(f'{f=}')
ro.set_injection_ctrl(500, 500)

prog = [
    *[ShiftIn16(16, True, 0x234) for _ in range(4096)],
    # *prog_sleep(int(f)), # 1 second
    Finish(),
]

cycles = count_cycles(prog)
print(f'{cycles=} {cycles/f=}')

ro.sm_write(prog)

############################################################################

tstart = time.perf_counter()

ro.sm_start(10, 0, 10)
ro.wait_sm_idle(20)

tend = time.perf_counter()
print(f'time: {tend-tstart}')

print(ro.get_comm_counters())

# n = 1 << 12
# while n > 0:
#     n -= 1

#     try:
#         # ro.read_register(ro.ADDR_SM_STATUS)
#         # print('ctrs: ', hex(ro.read_register(0xe0)))
#         print(n, hex(ro.read_register(ro.ADDR_SM_STATUS)), end='\r')
#     except Exception as e:
#         print(f'[red]>> {e}')
#         n = 10



