#!/usr/bin/env python
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

################################################################################


parser = argparse.ArgumentParser()

a_input_file = parser.add_argument(
    'input_file',
    nargs='+',
    help='h5 input files',
)

parser.add_argument(
    '--output',
    help='plot output file',
)

parser.add_argument(
    '--ymax',
    type=float, default=-1,
    help='maximum y value (microamps)',
)

try:
    import argcomplete
    from argcomplete.completers import FilesCompleter
    setattr(a_input_file, 'completer', FilesCompleter('txt'))
    argcomplete.autocomplete(parser)
except ImportError:
    pass

args = parser.parse_args()

################################################################################

for path_in in args.input_file:
    style = '--' if ('1.1' in path_in or '5e14' in path_in) else '-'
    path_in = Path(path_in)
    U, I = np.loadtxt(path_in, unpack=True)
    plt.plot(U, I * 1e6, style, label=path_in.name.removesuffix('-iv.txt'))
plt.legend()
plt.xlabel('Depletion Voltage (V)')
plt.ylabel('Leakage Current (µA)')

if args.ymax != 0:
    plt.ylim(bottom=-abs(args.ymax))


if args.output is not None:
    plt.savefig(args.output, dpi=300, transparent=True)
else:
    plt.show()
