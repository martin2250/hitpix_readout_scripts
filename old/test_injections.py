#!/usr/bin/python
from readout.instructions import *
import hitpix_roprog
import serial
import time
import datetime
import bitarray
import sys
from hitpix1 import HitPix1DacConfig
from hitpix1 import *
from readout.fast_readout import FastReadout

ro = HitPix1Readout('/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A6003YJ6-if00-port0')
ro.initialize()

ro.set_injection_ctrl(500, 500)

ro.set_treshold_voltage(1.3)
ro.set_baseline_voltage(1.1)
ro.set_injection_voltage(1.0)

# configure DAC
ro.sm_exec(
    hitpix_roprog.prog_dac_config(
        HitPix1DacConfig().generate(),
        4,
    ),
)

time.sleep(0.1)

##################################################################################

def prog_inject(num_injections: int, cfg_col: Union[HitPix1ColumnConfig, bitarray.bitarray]) -> list[Instruction]:
    cfg_int = SetCfg(
        shift_rx_invert = True,
        shift_tx_invert = True,
        shift_toggle = True,
        shift_select_dac = False,
        shift_word_len = 2 * 13,
        shift_clk_div = 2,
        pins = 0,
    )
    cfg_col_bit = cfg_col if isinstance(cfg_col, bitarray.bitarray) else cfg_col.generate()
    return [
        cfg_int.set_pin(HitPix1Pins.ro_rescnt, True),
        Sleep(10),
        cfg_int,
        Sleep(10),
        Reset(True, True),
        *prog_shift_dense(cfg_col_bit, False),
        Sleep(200),
        cfg_int.set_pin(HitPix1Pins.ro_ldconfig, True),
        Sleep(200),
        cfg_int,
        Sleep(500),
        cfg_int.set_pin(HitPix1Pins.ro_frame, True),
        Sleep(300),
        Inject(num_injections),
        Sleep(300),
        cfg_int,
        Sleep(300),
    ]

def prog_full_readout(shift_clk_div: int) -> list[Instruction]:
    cfg_int = SetCfg(
        shift_rx_invert = True,
        shift_tx_invert = True,
        shift_toggle = True,
        shift_select_dac = False,
        shift_word_len = 2 * 13,
        shift_clk_div = shift_clk_div,
        pins = 0,
    )

    prog: list[Instruction] = [
        cfg_int,
    ]

    for row in range(25):
        col_cfg = HitPix1ColumnConfig(0, 0, 0, row)
        # add time to make readout more consistent
        if row > 0:
            prog.append(GetTime())
        prog.extend([
            Sleep(100),
            Reset(True, True),
            *prog_shift_dense(col_cfg.generate(), row > 0),
            Sleep(100),
            cfg_int.set_pin(HitPix1Pins.ro_ldconfig, True),
            Sleep(100),
            cfg_int,
            Sleep(100),
        ])
        if row == 24:
            break
        prog.extend([
            # load count into column register
            cfg_int.set_pin(HitPix1Pins.ro_ldcnt, True),
            Sleep(100),
            cfg_int,
            Sleep(100),
            cfg_int.set_pin(HitPix1Pins.ro_penable, True),
            Sleep(100),
            ShiftOut(1, False),
            Sleep(100),
            cfg_int,
            Sleep(100),
        ])
    
    return prog




##################################################################################
# FULL READOUT

fastreadout = FastReadout('')

prog_readout = hitpix_roprog.prog_full_readout(1)
prog_readout.append(Finish())
ro.sm_write(prog_readout)

rows = 24

lines = []
responses = [fastreadout.expect_response() for _ in range(rows)]

for row in range(rows):
    col_cfg_inj = HitPix1ColumnConfig(
        (1 << row),
        (1 << 23),
        0,
        24,
    ).generate()
    ro.sm_exec(hitpix_roprog.prog_inject(100, col_cfg_inj))
    ro.sm_start()

    while True:
        status = ro.get_sm_status()
        if status.idle:
            break

for response in responses:
    response.event.wait(1.5)

for row, response in enumerate(responses):
    assert response.data is not None

    print('-'*30 + f'{row:4d}   ' + '-'*30)
    timestamps, hits = hitpix_roprog.decode_column_packets(response.data)
    for timestamp, hit, row_ro in zip(timestamps, hits, range(100)):
        hit = (256 - hit) % 256
        dt = datetime.datetime.fromtimestamp(ro.convert_time(timestamp)).time()
        if row == row_ro:
            dt = '---------------'
        print(f'{row_ro:2d} {dt}' + ' '.join(f'{h:3d}' for h in hit))
