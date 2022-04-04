#!/usr/bin/env python
'''
'''

import argparse
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np

################################################################################

parser = argparse.ArgumentParser()

a_input_file = parser.add_argument(
    'input_file',
    help='h5 input file',
)

parser.add_argument('--h5group', default='frames')
parser.add_argument('--mask', action='append')
parser.add_argument('--plot_type', choices=('hist_frame', 'hist_pixel', 'curve', 'hitmap'))
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
plot_type: str = args.plot_type
save: Optional[str] = args.save

mask = []
for mask_str in (args.mask or []):
    xy_str = mask_str.split(',')
    assert len(xy_str) == 2
    xy_index = []
    for s in xy_str:
        if s == ':':
            xy_index.append(slice(None))
            continue
        ss = list(map(int, s.split(':')))
        if len(ss) == 1:
            xy_index.append(ss[0])
        elif len(ss) == 2:
            xy_index.append(slice(*ss))
        else:
            raise Exception('not good')
    xy_index.append(...) # all frames -> ellipsis
    index = xy_index[::-1]
    print(f'masking {index}')
    mask.append(tuple(index))

################################################################################
# load data

print('loading data...', end='', flush=True)

with h5py.File(args.input_file) as file:
    # load frames group
    group_frames = file[h5group]
    assert isinstance(group_frames, h5py.Group)
    # load times dataset
    dset_frames = group_frames['frames']
    assert isinstance(dset_frames, h5py.Dataset)
    hits = dset_frames[()]
    assert isinstance(hits, np.ndarray)
    hits = np.flip(hits, axis=2)

for index in mask:
    hits[index] = 0

################################################################################

if plot_type == 'hitmap':
    plt.imshow(np.sum(hits, axis=0))
elif plot_type == 'hist_frame':
    hits_per_frame = np.sum(hits, axis=(1, 2))
    hits_per_frame = hits_per_frame[hits_per_frame != 0]
    plt.hist(hits_per_frame, 200)
elif plot_type == 'hist_pixel':
    hits_per_pixel = hits.flatten()
    hits_per_pixel = hits_per_pixel[hits_per_pixel != 0]
    plt.hist(hits_per_pixel, np.arange(20) + 0.5) # type: ignore
    plt.xticks(np.arange(20))
    plt.semilogy()
elif plot_type == 'curve':
    hits_per_frame = np.sum(hits, axis=(1, 2))
    plt.plot(hits_per_frame)

if args.save is None:
    plt.show()
else:
    plt.savefig(args.save, dpi=300)
