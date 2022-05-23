#!/usr/bin/env python
'''
'''

import argparse
from typing import Iterable, Optional, cast

import h5py
import matplotlib.pyplot as plt
import numpy as np
from rich import print
import sys
from pathlib import Path
import bottleneck as bn

if True:  # do not reorder with autopep8 or sortimports
    sys.path.insert(1, str(Path(__file__).parents[2]))
import frames.io
from util import parse_ndrange

################################################################################

parser = argparse.ArgumentParser()

a_input_file = parser.add_argument(
    'input_file',
    help='h5 input file',
)

parser.add_argument('--h5group', default='frames')
parser.add_argument('--mask', action='append')
parser.add_argument('--save')

################################################################################

try:
    import argcomplete
    from argcomplete.completers import FilesCompleter
    setattr(a_input_file, 'completer', FilesCompleter('h5'))
    argcomplete.autocomplete(parser)
except ImportError:
    pass

################################################################################

args = parser.parse_args()

input_file: str = args.input_file
h5group: str = args.h5group
save: Optional[str] = args.save

mask = [
    (...,) + parse_ndrange(mask_str, 2)
    for mask_str in (args.mask or [])
]

################################################################################
# load data

with h5py.File(args.input_file) as file:
    # load frames group
    group_frames = file[h5group]
    assert isinstance(group_frames, h5py.Group)
    # load frames dataset
    config, hits, times, time_sync = frames.io.load_frames(group_frames, load_times=True)

hits = np.flip(hits, axis=-1)

for index in mask:
    hits[index] = 0

################################################################################

plt.imshow(np.sum(hits, axis=0))


circles = [
    # HP2.19
    plt.Circle((25.5, 23), 3.5, color='r', fill=False),
    plt.Circle((25.5, 23), 6, color='r', fill=False),
]

ax = plt.gca()
for c in circles:
    ax.add_patch(c)

plt.show()

