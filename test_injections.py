#!/usr/bin/python
from statemachine import *
import hitpix_roprog
import serial
import time
import datetime
import bitarray
import sys
from hitpix1 import *
from readout import FastReadout, Response

ro = HitPix1Readout('/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A6003YJ6-if00-port0')
ro.initialize()

ro.set_injection_ctrl(255, 255)

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
# FULL READOUT

fastreadout = FastReadout()

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
