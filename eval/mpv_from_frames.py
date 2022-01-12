#!/usr/bin/env python

'''
this script should take in threshold voltage scans with Sr90 source and show the change in rate over the threshold voltage
in theory, the change in rate should follow the spectrum of the source
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
import util.gridscan

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    a_input_file = parser.add_argument(
        'input_file',
        help='h5 input file',
    )

    parser.add_argument('--show_pixel_selection', action='store_true')
    parser.add_argument('--skip', type=int, default=0)

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
        # get information about parameter scan
        assert 'scan' in file
        group_scan = file['scan']
        assert isinstance(group_scan, h5py.Group)
        scan_parameters, scan_shape = util.gridscan.load_scan(group_scan)

        # load first dataset to get data shape
        # old data format used 'scurve' prefix
        prefix = 'frames'
        if not (prefix + '_0' * len(scan_shape)) in file:
            prefix = 'scurve'
        group_frames = file[prefix + '_0' * len(scan_shape)]
        assert isinstance(group_frames, h5py.Group)
        config, hit_frames_first, _ = frames.io.load_frames(group_frames)
        # create full data array
        hits_frames = np.zeros(scan_shape + hit_frames_first.shape)
        # store first scurve
        hits_frames[tuple(0 for _ in scan_shape) + (...,)] = hit_frames_first
        # store remaining scurves
        for idx in np.ndindex(*scan_shape):
            # do not load zeroth frame again
            if not any(idx):
                continue
            group_name = prefix + '_' + '_'.join(str(i) for i in idx)
            group_frame = file[group_name]
            assert isinstance(group_frame, h5py.Group)
            _, hits_frames_group, _ = frames.io.load_frames(group_frame)
            hits_frames[idx] = hits_frames_group

    # only scan over threshold voltage for now
    assert len(scan_parameters) == 1
    x = scan_parameters[0].values

    ################################################################################
    # calculate pixel properties

    hits_frames = hits_frames.sum(axis=-3)

    # sort
    sort_x = np.argsort(x)
    x = x[sort_x]
    hits_frames = hits_frames[sort_x]

    # skip
    x = x[args.skip:]
    hits_frames = hits_frames[args.skip:]

    # convert to hits/s
    hits_frames = hits_frames / \
        (config.frame_length_us * config.num_frames * 1e-6)

    pixel_max = np.max(hits_frames, axis=0)
    pixel_sel = pixel_max < 2*np.percentile(pixel_max, 60)

    if args.show_pixel_selection:
        plt.imshow((pixel_max * pixel_sel).T)
        plt.show()

    rate_over_threshold = np.sum(
        hits_frames * pixel_sel.reshape(1, 24, 24), axis=(1, 2))

    plt.plot((x[:-1] + x[1:])*0.5, -np.diff(rate_over_threshold))
    plt.show()
