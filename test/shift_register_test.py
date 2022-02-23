#!/usr/bin/env python

from datetime import datetime
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
import numpy as np
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

# cfg_setup = 'hitpix1'
cfg_setup = 'hitpix2-1x1'

# cfg_clkdiv = 0
# cfg_clkdiv = 1
cfg_clkdiv = 2

cfg_select = 'ro'
# cfg_select = 'dac'

cfg_voltage = 1.85

# checks
assert cfg_clkdiv in range(3)
assert cfg_select in ['ro', 'dac']

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

# vdd_channel = util.voltage_channel.open_voltage_channel(
#     board.default_vdd_driver,
#     'VDD',
# )
# vdd_channel.set_voltage(cfg_voltage)

############################################################################
# readout program

def test_shift_register(
    ro: HitPixReadout,
    shift_clk_div: int,
    shift_sample_latency: int,
    test_string: bitarray.bitarray,
    num_rounds: int,
) -> bitarray.bitarray:
    cfg = SetCfg(
        shift_rx_invert = True,
        shift_tx_invert = False,
        shift_toggle = False,
        shift_select_dac = cfg_select == 'dac',
        shift_word_len = 32,
        shift_clk_div = shift_clk_div,
        shift_sample_latency=shift_sample_latency,
    )
    pins = SetPins(0)

    setup_registers = ro.setup.pixel_columns * ro.setup.chip.bits_adder

    if len(test_string) >= setup_registers:
        prog = [
            cfg,
            pins,
            Reset(True, True),
            Sleep(3),
            *prog_shift_dense(test_string[:setup_registers], False),
            *prog_shift_dense(test_string[setup_registers:], True),
            ShiftOut(setup_registers, True),
            *prog_sleep(int(10e3 * ro.frequency_mhz)),
        ]
    else:
        prog = [
            cfg,
            pins,
            Reset(True, True),
            Sleep(3),
            *prog_shift_dense(test_string, False),
            ShiftOut(setup_registers - len(test_string), False),
            ShiftOut(len(test_string), True),
            *prog_sleep(int(10e3 * ro.frequency_mhz)),
        ]

    ro.sm_write(prog + [Finish()])

    resp = fastreadout.expect_response()
    
    ############################################################################

    ro.sm_start(num_rounds)
    ro.wait_sm_idle(5.0)

    if not resp.event.wait(5):
        raise TimeoutError('no fastreadout response!')

    assert resp.data is not None
    data = np.frombuffer(resp.data, dtype=np.uint32).byteswap().tobytes()
    return bitarray.bitarray(buffer=data).copy()

################################################################################

# find all frequencies supported by readout
frequencies = []
f_last = 0
for f in np.linspace(10, 190, 30):
    f_new = ro.set_system_clock(f, dry_run=True)
    if f_new == f_last:
        continue
    f_last = f_new
    frequencies.append(float(f_new))

################################################################################

shift_latencies = list(range(0, 47))
test_string = bitarray.util.urandom(4*1024)

date = datetime.now().date().isoformat()
ro_version = ro.get_version()
filename = f'{date}-{cfg_setup}-{cfg_select}-v{ro_version.readout:03x}-div{cfg_clkdiv}-latency.dat'

with open(filename, 'w') as f_out:
    print(f'# latency scan', file=f_out)
    print(f'# {date=}', file=f_out)
    print(f'# {cfg_setup=}', file=f_out)
    print(f'# {cfg_voltage=}', file=f_out)
    print(f'# {cfg_clkdiv=}', file=f_out)
    print(f'# {cfg_select=}', file=f_out)
    print(f'# {ro_version=}', file=f_out)
    print(f'# latencies\t' + '\t'.join(str(int(l)) for l in shift_latencies), file=f_out)
    print(f'# frequency (MHz)\t' + '\t'.join(f'errors ({l:d})' for l in shift_latencies), file=f_out)
        
    for freq in frequencies:
        print(freq)
        ro.set_system_clock(freq)

        test_shift_register(
            ro=ro,
            shift_clk_div=cfg_clkdiv,
            shift_sample_latency=8,
            test_string=test_string,
            num_rounds=1,
        )

        error_counts = []

        for latency in shift_latencies:
            res = test_shift_register(
                ro=ro,
                shift_clk_div=cfg_clkdiv,
                shift_sample_latency=latency,
                test_string=test_string,
                num_rounds=8,
            )

            errors = 0
            while res:
                part, res = res[:len(test_string)], res[len(test_string):]
                diff = part ^ test_string
                errors += diff.count(1)

            print(f'{ro.frequency_mhz=:0.2f} {latency=:3d} {errors=}')

            error_counts.append(errors)
        line = f'{freq:0.3f}\t' + '\t'.join(str(ec) for ec in error_counts)
        print(line, file=f_out, flush=True)
        print(line)
