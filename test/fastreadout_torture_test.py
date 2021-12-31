#!/usr/bin/env python

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

cfg_words_per_prog = 2000 # less than 8kbytes per prog
cfg_prog_per_round = 5000 # 40MB per run -> approx 1s
cfg_rounds = 86400

cfg_len_expect = cfg_words_per_prog * cfg_prog_per_round * 4

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
    Sleep(4),
] * cfg_words_per_prog
prog.append(Finish())

ro.sm_write(prog)

############################################################################

n_tot = 0
t_start = time.perf_counter()
q_resp = queue.Queue()
running = True
error = False

def receive():
    global n_tot, running, q_resp, error
    while running or not q_resp.empty():
        resp: fast_readout.Response = q_resp.get()
        q_resp.task_done()
        resp.event.wait(5.0)
        if resp.data is None:
            print('data is None!')
            error = True
            continue
        if len(resp.data) != cfg_len_expect:
            print(f'data length {len(resp.data)} != {cfg_len_expect}')
            error = True
        n_tot += len(resp.data)


t_recv = threading.Thread(target=receive)
t_recv.start()

try:
    for _ in tqdm.tqdm(range(cfg_rounds), dynamic_ncols=True):
        q_resp.put(fastreadout.expect_response())
        ro.sm_start(cfg_prog_per_round)
        ro.wait_sm_idle(5.0)
        if error:
            print('aborting due to error')
            break
except KeyboardInterrupt:
    pass

running = False
t_recv.join()

t_end = time.perf_counter()
n_mb = n_tot / 1024**2
speed = n_mb / (t_end - t_start)
print(f'{n_mb/1024:0.2f} GB total, {speed:0.2f} MB/s')
