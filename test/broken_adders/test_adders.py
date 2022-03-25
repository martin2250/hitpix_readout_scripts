#!/usr/bin/env python3.10
'''
even without counting, the adder values seem to fluctuate

'''
import atexit
import copy
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path

import h5py
import numpy as np
from pandas import value_counts
import builtins
from rich import print
from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)

from rich.progress import track
if True:  # do not reorder with autopep8 or sortimports
    sys.path.insert(1, str(Path(__file__).parents[2]))

from readout.instructions import *
from readout.sm_prog import decode_column_packets, prog_dac_config, prog_shift_dense, prog_sleep
import hitpix
import hitpix.defaults
import util.configuration
import util.gridscan
import util.helpers
from hitpix import HitPixColumnConfig, ReadoutPins
from frames.daq import read_frames
from frames.io import FrameConfig, save_frame_attrs
from hitpix.readout import HitPixReadout
from readout.fast_readout import FastReadout
from readout.readout import SerialCobsComm
from util.live_view.frames import LiveViewAdders, LiveViewFrames
from util.voltage_channel import open_voltage_channel

############################################################################
# setup

# cfg_setup = 'hitpix1'
cfg_setup = 'hitpix2-1x1'
setup = hitpix.setups[cfg_setup]
chip = setup.chip

cfg_dac = chip.dac_config_class.default()
cfg_rows = list(range(setup.pixel_rows))

############################################################################
# default settings

cfg_output = sys.argv[1]
cfg_vddd = 1.95
cfg_vdda = 1.85
cfg_vssa = 1.25
cfg_baseline = 1.0
cfg_threshold = 1.1

cfg_injection_voltage = 1.3
cfg_injection_pulse_us = 2.5
cfg_injection_pause_us = 17.5

cfg_vddd_driver = 'default'
cfg_vdda_driver = 'default'
cfg_vssa_driver = 'default'

cfg_frequency = 40.0

cfg_inject_columns = 4
cfg_inject_num = 5
cfg_inject_total = 128

cfg_reset_readcount = 128

cfg_pulse_ns = 500.0

############################################################################
# user settings

cfg_vddd = 2.05

# cfg_inject_total = 50
# cfg_rows = list(range(10))
# cfg_rows = list(range(40,48))

############################################################################

pulse_cycles = math.ceil(cfg_pulse_ns * cfg_frequency / 1000)
column_pitch = setup.pixel_columns // cfg_inject_columns
assert column_pitch * cfg_inject_columns == setup.pixel_columns

if isinstance(cfg_dac, hitpix.HitPix2DacConfig):
    cfg_dac.vth = int(255 * cfg_threshold / cfg_vdda)

############################################################################
print('[yellow]connecting to readout')

config_readout = util.configuration.load_config()
serial_port_name, board = config_readout.find_board()

fastreadout = FastReadout(board.fastreadout_serial_number)
atexit.register(
    lambda: print('[yellow]closing fastreadout') or fastreadout.close(),
)

time.sleep(0.05)
ro = HitPixReadout(SerialCobsComm(serial_port_name), setup)
ro.initialize()
atexit.register(lambda: print('[yellow]closing readout') or ro.close())

print(f'[yellow]setting readout frequency to {cfg_frequency:0.1f} MHz')
ro.set_system_clock(cfg_frequency)
print(f'[green]readout frequency is {ro.frequency_mhz:0.1f} MHz')

############################################################################
print('[yellow]connecting to power supplies')

if cfg_vddd_driver == 'default':
    cfg_vddd_driver = board.default_vddd_driver
if cfg_vdda_driver == 'default':
    cfg_vdda_driver = board.default_vdda_driver
if cfg_vssa_driver == 'default':
    cfg_vssa_driver = board.default_vssa_driver

vddd_channel = open_voltage_channel(cfg_vddd_driver, 'VDDD')
vdda_channel = open_voltage_channel(cfg_vdda_driver, 'VDDA')
vssa_channel = open_voltage_channel(cfg_vssa_driver, 'VSSA')

print('[green]setting voltages')
vddd_channel.set_voltage(cfg_vddd)
vdda_channel.set_voltage(cfg_vdda)
vssa_channel.set_voltage(cfg_vssa)

ro.set_baseline_voltage(cfg_baseline)
ro.set_threshold_voltage(cfg_threshold)

ro.set_injection_voltage(cfg_injection_voltage)
ro.set_injection_ctrl(
    int(cfg_injection_pulse_us * ro.frequency_mhz),
    int(cfg_injection_pause_us * ro.frequency_mhz),
)

############################################################################
print('[yellow]configuring DACs')

ro.sm_exec(prog_dac_config(cfg_dac.generate()))

############################################################################

cfg = SetCfg(
    shift_rx_invert=True,
    shift_tx_invert=True,
    shift_toggle=True,
    shift_select_dac=False,
    shift_word_len=2 * chip.bits_adder,
    shift_clk_div=0,
    shift_sample_latency=setup.get_readout_latency(0, cfg_frequency),
)
pins = SetPins(0).set_pin(ReadoutPins.ro_psel, True)
pulse_sleep = prog_sleep(pulse_cycles - 1, cfg)

ctr_max = 1 << setup.chip.bits_counter
adder_max = ctr_max * setup.chip.rows

############################################################################
print('[yellow]reading adder output after rstcnt')

prog_reset_init = [
    cfg,
    pins,
    Reset(True, True),
    *prog_shift_dense(setup.encode_column_config(HitPixColumnConfig()), False),
    Sleep(3),
    # load registers
    *pins.pulse_pin(ReadoutPins.ro_ldconfig, True, pulse_sleep),
    # reset counters
    *pins.pulse_pin(ReadoutPins.ro_rescnt, True, pulse_sleep),
]

prog_reset = [
    cfg,
    pins,
    # load count into column register
    *pins.pulse_pin(ReadoutPins.ro_ldcnt, True, pulse_sleep),
    # shift one bit
    pins.set_pin(ReadoutPins.ro_penable, True),
    *pulse_sleep,
    ShiftOut(1, False),
    *prog_sleep(pulse_cycles + 3, cfg),,
    pins,
    GetTime(),
    Reset(True, True),
    *prog_shift_dense(setup.encode_column_config(HitPixColumnConfig()), True),
    Sleep(3),
    *pulse_sleep,
    *pins.pulse_pin(ReadoutPins.ro_ldconfig, True, pulse_sleep),
    Finish(),
]

ro.sm_exec(prog_reset_init)
resp = fastreadout.expect_response()
ro.sm_write(prog_reset)
ro.sm_start(cfg_reset_readcount)
ro.wait_sm_idle()
assert resp.event.wait(1.0)
assert resp.data

_, values_reset_raw = decode_column_packets(
    packet = resp.data,
    columns = setup.pixel_columns,
    bits_shift = chip.bits_adder,
    bits_mask = chip.bits_adder,
)
values_reset = np.zeros(setup.pixel_columns, dtype=np.int0)
for i in range(setup.pixel_columns):
    unique, counts = np.unique(
        values_reset_raw[:,i],
        return_counts=True,
    )
    # flip columns
    values_reset[-i] = unique[np.argmax(counts)]

print('[green]most common reset values:')
print(values_reset)

############################################################################

responses = [fastreadout.expect_response() for _ in cfg_rows]

for row in track(cfg_rows, description='injecting into rows'):
    # column configurations for all injection cycles
    col_cfgs = []
    for i in range(column_pitch):
        cols = 0
        for j in range(i, setup.pixel_columns, column_pitch):
            cols |= 1 << j
        col_cfgs.append(HitPixColumnConfig(
            1 << row,
            cols,
            0,  # ampout
            -1,  # readout
        ))
    col_cfgs_next = col_cfgs[1:] + col_cfgs[:1]
    # initialize
    prog_init = [
        cfg,
        pins,
        Reset(True, True),
        *prog_shift_dense(setup.encode_column_config(col_cfgs[0]), False),
        Sleep(3),
        # load registers
        *pins.pulse_pin(ReadoutPins.ro_ldconfig, True, pulse_sleep),
    ]
    # inject and read
    prog: list[Instruction] = []
    for col_cfg_next in col_cfgs_next:
        prog.extend([
            # reset counters
            *pins.pulse_pin(ReadoutPins.ro_rescnt, True, pulse_sleep),
            Sleep(1000),
            # inject
            pins.set_pin(ReadoutPins.ro_frame, True),
            Sleep(1000),
            Inject(cfg_inject_num),
            pins,
            # load count into column register
            *pins.pulse_pin(ReadoutPins.ro_ldcnt, True, pulse_sleep),
            # shift one bit
            pins.set_pin(ReadoutPins.ro_penable, True),
            *pulse_sleep,
            ShiftOut(1, False),
            *prog_sleep(pulse_cycles + 3, cfg),
            pins,
            # read shift register
            GetTime(),
            Reset(True, True),
            *prog_shift_dense(setup.encode_column_config(col_cfg_next), True),
            Sleep(3),
            *pulse_sleep,
            *pins.pulse_pin(ReadoutPins.ro_ldconfig, True, pulse_sleep),
        ])
    prog.append(Finish())

    ############################################################################

    ro.sm_exec(prog_init)
    ro.sm_write(prog)
    ro.sm_start(cfg_inject_total)
    ro.wait_sm_idle(10.0)

assert responses[-1].event.wait(1.0)

with open(cfg_output, 'w') as f_out:
    loc = dict(locals())
    for key, value in loc.items():
        if not key.startswith('cfg_'):
            continue
        builtins.print(f'# {key} = {repr(value)}', file=f_out)
    valres = '\t'.join(str(val) for val in values_reset)
    builtins.print(f'# values_reset = [{valres}]', file=f_out)
    for row, resp in zip(cfg_rows, responses):
        assert resp.data
        # decode binary
        _, values_raw = decode_column_packets(
            packet = resp.data,
            columns = setup.pixel_columns,
            bits_shift = chip.bits_adder,
            bits_mask = chip.bits_adder,
        )
        values_raw = values_raw.reshape(
            cfg_inject_total,
            column_pitch,
            setup.pixel_columns,
        )
        values_raw = np.flip(values_raw, axis=-1)
        hits_raw = values_reset[np.newaxis,np.newaxis,...] - values_raw
        # extract active columns
        hits = np.zeros((cfg_inject_total, setup.pixel_columns), dtype=np.int0)
        for i in range(column_pitch):
            hits[:,i::column_pitch] = hits_raw[:,i,i::column_pitch]
            hits_raw[:,i,i::column_pitch].fill(0)
        # how many single hits?
        hits_correct = hits == ((1 << chip.bits_counter) - cfg_inject_num)
        hits_row = np.sum(hits_correct, axis=0)
        print(f'[purple]row {row:2}[/purple]' + ''.join(f'{i:5}' for i in hits_row))
        builtins.print('\t'.join(str(i) for i in hits_row), file=f_out)
