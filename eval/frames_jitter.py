#!/usr/bin/env python
'''
'''

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Optional, cast

import h5py
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

if True:  # do not reorder with autopep8 or sortimports
    sys.path.insert(1, str(Path(__file__).parents[1]))
import frames.io

parser = argparse.ArgumentParser()

a_input_file = parser.add_argument(
    'input_file',
    help='h5 input file',
)

parser.add_argument('--h5group', default='frames')
parser.add_argument('--histogram', action='store_true')
parser.add_argument('--save')

try:
    import argcomplete
    from argcomplete.completers import FilesCompleter
    setattr(a_input_file, 'completer', FilesCompleter('h5'))
    argcomplete.autocomplete(parser)
except ImportError:
    pass


args = parser.parse_args()

input_file: str = args.input_file
h5group: str = args.h5group
histogram: bool = args.histogram
save: Optional[str] = args.save

################################################################################
# load data

print('loading data...', end='', flush=True)

with h5py.File(args.input_file) as file:
    # load frames group
    group_frames = file[h5group]
    assert isinstance(group_frames, h5py.Group)
    # load times dataset
    dset_times = group_frames['times']
    assert isinstance(dset_times, h5py.Dataset)
    times = dset_times[()]
    assert isinstance(times, np.ndarray)

# microseconds
time_diff = np.diff(times).astype(np.int32)

tmin = np.min(time_diff)
tmax = np.max(time_diff)

if (tmax - tmin) < 100:
    bins = np.arange(math.floor(tmin - 5), math.ceil(tmax + 5))
else:
    bins = np.geomspace(tmin, tmax, 100)

plt.hist(time_diff, cast(Any, bins))
plt.semilogy()
plt.semilogx()
plt.xlabel('time interval (µs)')
plt.show()
