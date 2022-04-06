#!/usr/bin/env python
'''
ffmpeg -framerate 30 -pattern_type glob -i '*.jpg' test.mp4
'''

import argparse
import time
from typing import Optional

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
import scipy.ndimage
from concurrent.futures import ProcessPoolExecutor

if True:  # do not reorder with autopep8 or sortimports
    sys.path.insert(1, str(Path(__file__).parents[2]))
from util import parse_ndrange

################################################################################

parser = argparse.ArgumentParser()

a_input_file = parser.add_argument(
    'input_file',
    help='h5 input file',
)

parser.add_argument('--plot_average', action='store_true')
parser.add_argument('--plot_largest', type=int)
parser.add_argument('--plot_larger', type=int)
parser.add_argument('--plot_all', action='store_true')

parser.add_argument('--diagonal', action='store_true')
parser.add_argument('--h5group', default='frames')
parser.add_argument('--mask', action='append')
parser.add_argument('--save', help='output image file, use {} as placeholder')

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
diagonal: bool = args.diagonal
plot_average: bool = args.plot_average
plot_all: bool = args.plot_all
plot_largest: Optional[int] = args.plot_largest
plot_larger: Optional[int] = args.plot_larger

mask = [
    (...,) + parse_ndrange(mask_str, 2)
    for mask_str in (args.mask or [])
]

del args

def show_save(name: str):
    if save is None:
        plt.title(name)
        plt.show()
    else:
        plt.savefig(save.replace('{}', name))

################################################################################
# load data

print('loading data')

with h5py.File(input_file) as file:
    # load frames group
    group_frames = file[h5group]
    assert isinstance(group_frames, h5py.Group)
    # load frames dataset
    dset = group_frames['frames']
    assert isinstance(dset, h5py.Dataset)
    hits = dset.astype(np.uint32)[()] # 8 bits is enough for counters
    assert isinstance(hits, np.ndarray)
    # flip hitmap along x
    hits = np.flip(hits, axis=-1)

# apply mask
for index in mask:
    hits[index] = 0

################################################################################

# only allow connections inside a single frame, not between successive frames
structure = np.zeros((3, 3, 3), dtype=np.bool8)
if diagonal:
    structure[1] = True
else:
    # generates plus shaped mask
    structure[1, 1, :] = True
    structure[1, :, 1] = True

print('labelling clusters')
labels = hits # rename
# replace hits with cluster labels
scipy.ndimage.label(hits, structure=structure, output=labels)

print('measuring cluster sizes')
# measure size of each label
sizes = np.bincount(labels.flat)
# do not replace zero with anything
sizes[0] = 0

print('mapping size to frames')
# replace labels by their sizes
cluster_sizes = sizes[labels]

# calculate mean cluster size
cluster_sum = np.sum(cluster_sizes, axis=0)
cluster_count = np.sum(cluster_sizes != 0, axis=0)
cluster_mean = cluster_sum.astype(np.float32) / cluster_count
cluster_max = np.max(cluster_sizes, axis=(1, 2))

################################################################################

if plot_average:
    plt.suptitle('Average Cluster Size (Pixels)')
    plt.imshow(cluster_mean)
    plt.colorbar()
    show_save('average')

if not (plot_all or (plot_larger is not None) or (plot_largest is not None)):
    exit()

norm = matplotlib.colors.Normalize(1, np.max(cluster_max))

def plot_process(suptitle: str, frame_id: int, filename: str):
    plt.clf()
    plt.suptitle(suptitle)
    frame = cluster_sizes[frame_id]
    # frame = np.where(frame == 0, np.nan, frame)
    plt.imshow(frame, norm=norm)
    plt.colorbar()
    show_save(filename)

with ProcessPoolExecutor() as pool:
    futures = []

    print('queueing plots')

    if plot_all:
        for frame_index in range(len(cluster_sizes)):
            futures.append(pool.submit(
                plot_process,
                f'Frame #{frame_index}',
                frame_index,
                f'all_{frame_index:06d}'
            ))

    if plot_larger is not None:
        frame_indices, = np.where(cluster_max > plot_larger)
        for file_index, frame_index in enumerate(frame_indices):
            size = np.max(cluster_max[frame_index])
            futures.append(pool.submit(
                plot_process,
                f'Cluster: {size} Pixels',
                frame_index,
                f'larger_{file_index:06d}'
            ))

    if plot_largest is not None:
        frame_indices = np.argsort(-cluster_max)
        frame_indices = frame_indices[:plot_largest]
        for file_index, frame_index in enumerate(frame_indices):
            size = np.max(cluster_max[frame_index])
            futures.append(pool.submit(
                plot_process,
                f'Cluster: {size} Pixels',
                frame_index,
                f'largest_{file_index:06d}'
            ))

    print('waiting for plots to finish')

    while futures:
        futures = list(filter(lambda f: not f.done(), futures))
        print(f'{len(futures):4d} plots remaining', end='\r')
        time.sleep(0.1)
    print('')
    print('done')