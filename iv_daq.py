#!/usr/bin/env python
import argparse
import math
import time
import util.gridscan
import util.configuration
import util.voltage_channel
import numpy as np
from pathlib import Path
import atexit
from pymeasure.instruments.keithley import Keithley2400

################################################################################

parser = argparse.ArgumentParser()

parser.add_argument(
    '--voltage_max',
    type=float, default=120.0,
    help='max HV (positve number -> negative voltage)',
)

parser.add_argument(
    '--voltage_step',
    type=float, default=1.0,
    help='voltage step (positve number -> negative voltage step)',
)

parser.add_argument(
    '--compliance',
    default=105.0, type=float,
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

compliance_uA = float(args.compliance)
output_file = Path(args.output_file)
voltage_max = -1.0 * float(args.voltage_max)
voltage_step = -1.0 * float(args.voltage_step)

assert voltage_max < 0
assert voltage_step < 0

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

hv_channel = util.voltage_channel.open_voltage_channel(board.default_hv_driver, 'HV')
assert isinstance(hv_channel, util.voltage_channel.Keithley2400VoltageChannel)
smu = hv_channel.smu
assert isinstance(smu, Keithley2400)

################################################################################

num_triggers = 1 + math.ceil(voltage_max / voltage_step)

# https://gist.github.com/cfangmeier/1859de7d67ab74099a3b#file-keithley2410_sweeptest-py-L22
smu.reset()
smu.measure_concurent_functions = False
smu.apply_voltage(voltage_range=voltage_max,compliance_current=compliance_uA*1e-6)
smu.measure_current(nplc=5, current=-compliance_uA*1e-6, auto_range=False)
smu.use_front_terminals()

smu.write(f':SOUR:VOLT:START 0.0')
smu.write(f':SOUR:VOLT:STOP {voltage_max:0.2f}')
smu.write(f':SOUR:VOLT:STEP {voltage_step:0.2f}')
smu.write(f':SOUR:VOLT:MODE SWE')
smu.write(f':SOUR:SWE:RANG AUTO')
smu.write(f':SOUR:SWE:SPAC LIN')
smu.write(f':SOUR:SWE:CAB LATE') # compliance abort
smu.write(f':TRIG:COUN {num_triggers}')
smu.write(f':SOUR:DEL 0.5')
smu.write(f':FORM:ELEM VOLT,CURR')


smu.enable_source()

smu.write(':READ?')

try:
    with open(output_file, 'w') as f_out:
        print('# IV Curve', file=f_out)
        print('# Voltage (V)\tCurrent (A)', file=f_out)
        while num_triggers > 0:
            data = str(smu.read())
            if not data:
                continue
            for line in data.splitlines():
                line = line.strip()
                line = line.strip(',')
                if not line:
                    continue
                try:
                    voltage, current = map(float, line.split(','))
                except:
                    print(f'error parsing line {line:r}')
                print(f'{voltage:6.2f} V  {1e6*current:8.2f} µA')
                print(f'{voltage:e}\t{current:e}', file=f_out)
                num_triggers -= 1
    for _ in range(5):
        smu.beep(3000, 0.2)
        time.sleep(0.3)
except KeyboardInterrupt:
    pass


smu.shutdown()
