#!/usr/bin/env python
import argparse
import time
import util.gridscan
import util.configuration
import util.voltage_channel
import numpy as np
from pathlib import Path
import atexit

################################################################################

parser = argparse.ArgumentParser()

parser.add_argument(
    '--voltages',
    default='0:60.0:100',
    help='voltages to measure (positive, automatically converted to negative)',
)

parser.add_argument(
    '--compliance',
    default=100.0, type=float,
    help='compliance current, µA',
)

parser.add_argument(
    'output_file',
    help='output text file',
)

try:
    import argcomplete
    argcomplete.autocomplete(parser)
except ImportError:
    pass

args = parser.parse_args()

################################################################################

voltages = util.gridscan.parse_range(args.voltages)
compliance_uA = float(args.compliance)
output_file = Path(args.output_file)

# IMPORTANT!!!
voltages = voltages * -1.0
assert not any(x > 0 for x in voltages)

################################################################################

if output_file.exists():
    try:
        res = input(
            f'file {output_file} exists, [d]elete or [N] abort? (d/N): ')
    except KeyboardInterrupt:
        exit()
    if res.lower() == 'd':
        output_file.unlink()
    else:
        exit()
        
################################################################################

config_readout = util.configuration.load_config()
serial_port_name, board = config_readout.find_board()

hv_channel = util.voltage_channel.open_voltage_channel(board.default_hv_driver, 'VDD')
assert isinstance(hv_channel, util.voltage_channel.Keithley2400VoltageChannel)
smu = hv_channel.smu

################################################################################

smu.source_voltage = 0.0

smu.apply_voltage(compliance_current=compliance_uA*1e-6)
smu.auto_range_source()
smu.enable_source()

atexit.register(smu.shutdown)

################################################################################

with open(output_file, 'w') as f_out:
    print('# IV Curve', file=f_out)
    print('# Voltage (V)\tCurrent (A)', file=f_out)

    for i, voltage in enumerate(voltages):
        print(f'set {voltage:0.2f}V', end=' ')
        smu.source_voltage = voltage

        smu.measure_voltage()
        for _ in range(10):
            voltage_meas = smu.voltage
            print(f'is {voltage_meas:0.3f}V', end=' ')
            if abs(voltage_meas - voltage) < 0.1:
                break
            time.sleep(0.05)
        else:
            print('voltage not reached, aborting')
            print('# voltage not reached, aborting', file=f_out)
            break # proably ran into compliance

        smu.measure_current()
        current_meas = smu.current

        print(f'{voltage_meas:e}\t{current_meas:e}', file=f_out)
        print(f'is {current_meas*1e6:0.4f}µA')

