#!/usr/bin/python
from readout import *
from statemachine import *
import hitpix1_config
import hitpix_roprog
import serial
import time
import datetime
import bitarray
import sys

readout = Readout('/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A6003YJ6-if00-port0')
readout.initialize()

readout.set_injection_ctrl(255, 255)
readout.write_register(
    readout.ADDR_SM_INVERT_PINS,
    (1 << hitpix_roprog.HitPix1Pins.ro_ldconfig) | (1 << hitpix_roprog.HitPix1Pins.dac_ld) | (1 << 30) | (1 << 31)
)

voltage_card = DacCard(0, 8, 3.3, 1.8, readout)
injection_card = DacCard(2, 2, 3.3, 1.8, readout)

voltage_card.set_voltage(hitpix_roprog.HitPix1VoltageCards.threshold, 1.3, False)
voltage_card.set_voltage(hitpix_roprog.HitPix1VoltageCards.baseline, 1.1)
injection_card.set_voltage(0, 1)

daccfg = hitpix1_config.DacConfig(
    q0 = 0x01,
    qon = 0x05,
    blres = 63,
    vn1 = 30,
    vnfb = 63,
    vnfoll = 4,
    vndell = 8,
    vn2 = 0,
    vnbias = 4,
    vpload = 3,
    vncomp = 5,
    vpfoll = 7,
)


prog_dac_cfg = hitpix_roprog.prog_dac_config(daccfg, 4)
prog_dac_cfg.append(Finish())
readout.sm_write([instr.to_binary() for instr in prog_dac_cfg])
readout.sm_start()

time.sleep(0.2)

##################################################################################
# FULL READOUT


fastreadout = FastReadout()

lines = []

for row in range(24):
    lines.append('-'*30 + f'{row:4d}   ' + '-'*30)
    readout.sm_exec(hitpix_roprog.prog_rescnt())

    col_cfg_inj = hitpix1_config.ColumnConfig(
        (1 << row),
        (1 << 23),
        0,
        24,
    ).generate()

    prog_inject = hitpix_roprog.prog_inject(100, col_cfg_inj)
    prog_inject.append(Finish())
    readout.sm_write(prog_inject)
    readout.sm_start()

    response = fastreadout.expect_response()

    prog_readout = hitpix_roprog.prog_full_readout(1)
    prog_readout.append(Finish())
    readout.sm_write(prog_readout)
    readout.sm_start()

    response.event.wait(1.5)
    assert response.data is not None

    timestamps, hits = hitpix_roprog.decode_column_packets(response.data)
    # assert len(hits) == 48
    # hits = hits[:24] + hits[24:]
    for timestamp, hit, row_ro in zip(timestamps, hits, range(100)):
        hit = (256 - hit) % 256
        dt = datetime.datetime.fromtimestamp(readout.convert_time(timestamp)).time()
        if row == row_ro:
            dt = '---------------'
        lines.append(f'{row_ro:2d} {dt}' + ' '.join(f'{h:3d}' for h in hit))

for line in lines:
    print(line, flush=False)
print('------')

##################################################################################
# INJETCIONS

# test_injection = hitpix_roprog.TestInjection(30, 1)
# prog_injection = test_injection.prog_test()

# readout.sm_write([instr.to_binary() for instr in prog_injection])

# # only works with 1-4 right now?
# readout.sm_start(1)

##################################################################################


