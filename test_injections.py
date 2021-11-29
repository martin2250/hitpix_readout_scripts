#!/usr/bin/python
from readout import *
from statemachine import *
import hitpix1_config
import hitpix_roprog
import serial
import time
import datetime
import bitarray

port = serial.Serial('/dev/ttyUSB0', 3000000)
readout = Readout(port)
readout.initialize()

readout.set_injection_ctrl(255, 255)
readout.write_register(
    readout.ADDR_SM_INVERT_PINS,
    (1 << hitpix_roprog.HitPix1Pins.ro_ldconfig) | (1 << hitpix_roprog.HitPix1Pins.dac_ld) | (1 << 30) | (1 << 31)
)

voltage_card = DacCard(0, 8, 3.3, 1.8, readout)
injection_card = DacCard(2, 2, 3.3, 1.8, readout)

fastreadout = FastReadout()

daccfg = hitpix1_config.DacConfig(
    q0 = 0x01,
    qon = 0x05,
    blres = 63,
    vn1 = 33,
    vnfb = 63,
    vnfoll = 4,
    vndell = 8,
    vn2 = 0,
    vnbias = 0,
    vpload = 1,
    vncomp = 5,
    vpfoll = 7,
)

voltage_card.set_voltage(hitpix_roprog.HitPix1VoltageCards.threshold, 1.4, False)
voltage_card.set_voltage(hitpix_roprog.HitPix1VoltageCards.baseline, 1.25)
injection_card.set_voltage(0, 1.4)

readout.sm_exec([instr.to_binary() for instr in hitpix_roprog.prog_dac_config(daccfg)])



test_injection = hitpix_roprog.TestInjection(30, 7)
prog_injection = test_injection.prog_test()

readout.sm_write([instr.to_binary() for instr in prog_injection])

# only works with 1-4 right now?
readout.sm_start(1)

time.sleep(0.2)

for packet in fastreadout.packets:
    timestamps, hits = hitpix_roprog.decode_column_packets(packet)
    for timestamp, hit in zip(timestamps, hits):
        dt = datetime.datetime.fromtimestamp(readout.convert_time(timestamp))
        print(dt.time(), list(hit))
