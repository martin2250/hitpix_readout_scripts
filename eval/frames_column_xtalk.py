#!/usr/bin/env python
'''
this script takes frame recordings (do not use --sums_only!) and checks which columns appear noisy
'''

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

if True:  # do not reorder with autopep8 or sortimports
    sys.path.insert(1, str(Path(__file__).parents[1]))
import frames.io

is_interactive = 'ipykernel' in sys.argv[0]

parser = argparse.ArgumentParser()

a_input_file = parser.add_argument(
    'input_file',
    help='h5 input file',
)

parser.add_argument('--hits_per_frame', action='store_true')
parser.add_argument('--multi_hits_per_col', action='store_true')
parser.add_argument('--hits_per_column', action='store_true')
parser.add_argument('--clean_hits', action='store_true')
parser.add_argument('--save')

try:
    import argcomplete
    from argcomplete.completers import FilesCompleter
    setattr(a_input_file, 'completer', FilesCompleter('h5'))
    argcomplete.autocomplete(parser)
except ImportError:
    pass

if is_interactive:
    args = parser.parse_args(['/home/martin/Desktop/master-thesis/hitpix_readout/data/2022-01-19/hitpix2/sr90/frames_5us.h5'])
else:
    args = parser.parse_args()

################################################################################
# load data

print('loading data...', end='', flush=True)

with h5py.File(args.input_file) as file:
    group_frames = file['frames']
    assert isinstance(group_frames, h5py.Group)
    config, hits, _ = frames.io.load_frames(group_frames, load_times=False)

print(' done')

assert hits.ndim == 3
hits = np.flip(hits, axis=1).astype(np.int32)

hits_per_col = np.sum(hits, axis=1)
hits_per_frame = np.sum(hits_per_col, axis=1)
col_hit_idx = np.argwhere(hits_per_col > 0) # which columns have been hit?

if args.hits_per_frame:
    plt.hist(
        hits_per_frame,
        bins=np.arange(180+1)-0.5,
        density=True,
        log=True,
    )
    plt.xlabel(f'Hits per frame ({config.frame_length_us:0.1f} us)')
    plt.ylabel('Fraction of Frames')

if args.multi_hits_per_col:
    multi_hits = hits_per_col >= 30
    plt.bar(
        np.arange(48),
        np.sum(multi_hits, axis=0),
        log=True,
    )
    plt.xlabel('Column Index')
    plt.ylabel('Full Column Hits')

if args.hits_per_column:
    plt.hist2d(
        x=col_hit_idx[:,1],
        y=hits_per_col[col_hit_idx[:,0], col_hit_idx[:,1]],
        bins=(
            np.arange(48+1)-0.5,
            np.arange(101)-0.5,
        ),
        norm=matplotlib.colors.LogNorm(),
    )
    plt.xlabel('Column Index')
    plt.ylabel('Number of Hits per Frame')

if args.clean_hits:
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(
        np.sum(hits, axis=0),
        norm=matplotlib.colors.LogNorm(),
    )
    ax1.set_title('Raw')

    full_column_hits = hits_per_col >= 48
    newshape = list(full_column_hits.shape)
    newshape.insert(-1, 1)
    hits = hits - np.reshape(full_column_hits, tuple(newshape))

    ax2.imshow(
        np.sum(hits, axis=0),
        norm=matplotlib.colors.LogNorm(),
    )
    ax2.set_title('Removed full Columns')

if not is_interactive:
    if args.save is not None:
        plt.savefig(args.save, dpi=300, transparent=False)
    else:
        plt.show()
