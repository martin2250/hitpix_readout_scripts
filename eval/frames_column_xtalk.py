#!/usr/bin/env python

'''
this script takes frame recordings (do not use --sums_only!) and checks which columns appear noisy
'''

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

if True:  # do not reorder with autopep8 or sortimports
    sys.path.insert(1, str(Path(__file__).parents[1]))

import frames.io

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    a_input_file = parser.add_argument(
        'input_file',
        help='h5 input file',
    )

    try:
        import argcomplete
        from argcomplete.completers import FilesCompleter
        setattr(a_input_file, 'completer', FilesCompleter('h5'))
        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args()

    ################################################################################
    # load data

    with h5py.File(args.input_file) as file:
        group_frames = file['frames']
        assert isinstance(group_frames, h5py.Group)
        config, hits, _ = frames.io.load_frames(group_frames)
    
    assert hits.ndim == 3
    
    hits_per_col = np.sum(hits, axis=1)
    hits_per_frame = np.sum(hits, axis=(1, 2))

    plt.hist(hits_per_frame, 100)
    plt.show()

    col_hit_idx = np.argwhere(hits_per_col > 0)

    print(f'{col_hit_idx=}')
    print(f'{len(col_hit_idx)=}')