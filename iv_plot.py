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
    path_in = Path(path_in)
    U, I = np.loadtxt(path_in, unpack=True)
    plt.plot(U, I, label=path_in.name)
plt.legend()
plt.show()