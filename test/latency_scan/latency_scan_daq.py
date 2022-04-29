#!/usr/bin/env python

from datetime import datetime
import sys
from pathlib import Path


if True:  # do not reorder with autopep8 or sortimports
    sys.path.insert(1, str(Path(__file__).parents[2]))

import atexit
import time
import bitarray.util
import bitarray
import numpy as np
import util.configuration
from readout.sm_prog import prog_shift_dense, prog_sleep
import util.gridscan
import util.voltage_channel
from hitpix.readout import HitPixReadout
from readout.readout import SerialCobsComm
import hitpix
import hitpix.defaults
from readout.fast_readout import FastReadout
from readout.instructions import Finish, Reset, SetCfg, SetPins, ShiftOut, Sleep

############################################################################

# cfg_setup = 'hitpix1'
# cfg_setup = 'hitpix2-1x1'
cfg_setup = 'hitpix2-1x5'

cfg_div1_clk1 = hitpix.setups[cfg_setup].readout_div1_clk1
cfg_div1_clk2 = hitpix.setups[cfg_setup].readout_div1_clk2

cfg_clkdiv_write = 0
# cfg_clkdiv_write = 1
# cfg_clkdiv_write = 2
# cfg_clkdiv_write = 3

cfg_clkdiv_read = None
# cfg_clkdiv_read = 3

cfg_shift_latencies = list(range(4, 27))


cfg_select = 'ro'
# cfg_select = 'dac'

# # hitpix 1 default
# cfg_div1_dtin=0b111111000000
# cfg_div1_clk1=0b001110000000
# cfg_div1_clk2=0b000001110000

# # hitpix 2 default
# cfg_div1_dtin=0b111111000000
# cfg_div1_clk1=0b001100000000
# cfg_div1_clk2=0b000001100000

# cfg_voltage = 1.85
cfg_voltage = None


cfg_num_rounds = 10
# cfg_num_rounds = 20

cfg_freq_range = np.linspace(16, 35, 30)

# checks
assert cfg_clkdiv_write in range(4)
assert cfg_select in ['ro', 'dac']

############################################################################
# open readout

config_readout = util.configuration.load_config()
serial_port_name, board = config_readout.find_board()

fastreadout = FastReadout(board.fastreadout_serial_number)
atexit.register(fastreadout.close)
time.sleep(0.05)

ro = HitPixReadout(SerialCobsComm(serial_port_name), hitpix.setups[cfg_setup])
ro.initialize()
atexit.register(ro.close)

if cfg_voltage is not None:
    vdd_channel = util.voltage_channel.open_voltage_channel(
        board.default_vddd_driver,
        'VDDD',
    )
    vdd_channel.set_voltage(cfg_voltage)

############################################################################
# precalculate stuff

if cfg_select == 'ro':
    # readout
    setup_registers = ro.setup.pixel_columns * ro.setup.chip.bits_adder
else:
    # dac, has slow outputs
    setup_registers = ro.setup.chip_columns * len(ro.setup.chip.dac_config_class.default().generate())

test_string = bitarray.util.urandom(setup_registers)

if cfg_clkdiv_read is None:
    if cfg_select == 'ro':
        cfg_clkdiv_read = cfg_clkdiv_write
    else:
        cfg_clkdiv_read = 3

# no need for different latencies with slow debug readout
if cfg_clkdiv_read == 3:
    cfg_shift_latencies = [0]

############################################################################
# readout program


def test_shift_register(
    ro: HitPixReadout,
    shift_clk_div_write: int,
    shift_clk_div_read: int,
    shift_sample_latency: int,
) -> list[bitarray.bitarray]:
    read_words, read_remaining = setup_registers // 32, setup_registers % 32

    # config instructions
    pins = SetPins(0)
    cfg_write = SetCfg(
        shift_rx_invert=True,
        shift_tx_invert=cfg_select == 'dac',
        shift_toggle=False,
        shift_select_dac=cfg_select == 'dac',
        shift_word_len=32,
        shift_clk_div=shift_clk_div_write,
        shift_sample_latency=shift_sample_latency,
    )
    cfg_read = cfg_write.modify(shift_clk_div=shift_clk_div_read)
    cfg_clear = cfg_write.modify(shift_clk_div=3)  # clear shift register

    # sm prog
    prog = [
        pins,
        Reset(True, True),
        # shift in a bunch of zeros
        cfg_clear,
        ShiftOut(setup_registers, False),
        # shift in actual data
        cfg_write,
        Sleep(5),
        *prog_shift_dense(test_string, False),
        Sleep(3),
        # start reading
        cfg_read,
        Reset(True, True),
        ShiftOut(read_words*32, True),
    ]
    if read_remaining > 0:
        prog += [
            # wait for previous read to finish (takes 64 clock cycles with clkdiv 3)
            Sleep(64),
            cfg_read.modify(shift_word_len=read_remaining),
            ShiftOut(read_remaining, True),
        ]
    prog += [
        Sleep(64),
        *prog_sleep(int(100e-6 * ro.frequency_mhz)),
        Finish(),
    ]

    ro.sm_write(prog)

    resp = fastreadout.expect_response()

    ############################################################################

    ro.sm_start(cfg_num_rounds)
    ro.wait_sm_idle(5.0)

    if not resp.event.wait(5):
        raise TimeoutError('no fastreadout response!')

    assert resp.data is not None
    data = np.frombuffer(resp.data, dtype=np.uint32)

    data = data.reshape((
        cfg_num_rounds,
        read_words + (read_remaining > 0),
    ))

    result = []
    for subdata in data:
        buffer = subdata.byteswap().tobytes()
        ba = bitarray.bitarray(buffer=buffer)
        ba_res = ba[:32*read_words]
        if read_remaining > 0:
            ba_res += ba[-read_remaining:]
        result.append(ba_res.copy())
    return result

################################################################################


# find all frequencies supported by readout
frequencies = []
f_last = 0
for f in cfg_freq_range:
    f_new = ro.set_system_clock(f, dry_run=True)
    if f_new == f_last:
        continue
    f_last = f_new
    frequencies.append(float(f_new))

################################################################################

date = datetime.now().date().isoformat()
ro_version = ro.get_version()
filename = f'{date}-{cfg_setup}-{cfg_select}-v{ro_version.readout:03x}-div{cfg_clkdiv_write}-cks-{cfg_div1_clk1}-{cfg_div1_clk2}-latency.dat'

with open(filename, 'w') as f_out:
    print(f'# latency scan', file=f_out)
    print(f'# {date=}', file=f_out)
    print(f'# {cfg_setup=}', file=f_out)
    print(f'# {cfg_voltage=}', file=f_out)
    print(f'# {cfg_clkdiv_write=}', file=f_out)
    print(f'# {cfg_clkdiv_read=}', file=f_out)
    print(f'# {cfg_select=}', file=f_out)
    print(f'# {ro_version=}', file=f_out)
    print(f'# {cfg_div1_clk1=:12b}', file=f_out)
    print(f'# {cfg_div1_clk2=:12b}', file=f_out)
    print(f'# latencies\t' + '\t'.join(str(int(l))
          for l in cfg_shift_latencies), file=f_out)
    print(f'# frequency (MHz)\t' +
          '\t'.join(f'errors ({l:d})' for l in cfg_shift_latencies), file=f_out)

    for freq in frequencies:
        ro.set_system_clock(freq)

        ro.set_readout_clock_sequence(cfg_div1_clk1, cfg_div1_clk2)

        test_shift_register(
            ro=ro,
            shift_clk_div_write=cfg_clkdiv_write,
            shift_clk_div_read=cfg_clkdiv_read,
            shift_sample_latency=8,
        )

        error_counts = []

        for latency in cfg_shift_latencies:
            res = test_shift_register(
                ro=ro,
                shift_clk_div_write=cfg_clkdiv_write,
                shift_clk_div_read=cfg_clkdiv_read,
                shift_sample_latency=latency,
            )

            errors = 0
            for r in res:
                diff = r ^ test_string
                diff_sum = diff.count(1)
                errors += diff_sum
                # if diff_sum > 0:
                #     print(r[:60])
                #     print(test_string[:60])

            print(f'{ro.frequency_mhz=:0.2f} {latency=:3d} {errors=}')

            error_counts.append(errors)
        line = f'{freq:0.3f}\t' + '\t'.join(str(ec) for ec in error_counts)
        print(line, file=f_out, flush=True)
        print(line)
