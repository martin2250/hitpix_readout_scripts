#!/usr/bin/env python
'''
'''

import argparse
import json
from typing import Iterable, Optional, cast

import h5py
import matplotlib.pyplot as plt
import numpy as np
# from rich import print
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
parser.add_argument('--ion', choices=('proton', 'carbon'), required=True)
parser.add_argument('--plot_type', choices=('curve', 'fit_curves', 'fit_scatter'))
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
ion: str = args.ion
plot_type: str = args.plot_type
h5group: str = args.h5group
save: Optional[str] = args.save

mask = [
    (...,) + parse_ndrange(mask_str, 1)
    for mask_str in (args.mask or [])
]

def plot():
    if save is not None:
        plt.savefig(save, dpi=300, transparent=True)
    else:
        plt.show()

################################################################################
# load data

with h5py.File(args.input_file) as file:
    # load frames group
    group_frames = file[h5group]
    assert isinstance(group_frames, h5py.Group)
    # load frames dataset
    config_dict = json.loads(cast(str, group_frames.attrs['config']))

    dset_adders = group_frames['adders']
    assert isinstance(dset_adders, h5py.Dataset)
    hits = dset_adders[()]
    assert isinstance(hits, np.ndarray)

hits = np.flip(hits, axis=1)
for index in mask:
    hits[index] = 0

################################################################################

# subtract constant offset
hits = hits - hits[0][np.newaxis, :]

def hits_per_frame_nonsaturated(hits: np.ndarray, quantile: float) -> np.ndarray:
    '''find and sum up the pixels with the fewest hits
    -> probably not in analog saturation, can be used to group spills by intensity'''
    pixel_total = np.sum(hits, axis=0)
    idx_ok = pixel_total < np.quantile(pixel_total, quantile)
    return np.sum(hits[..., idx_ok], axis=1)

def find_spills(
    hits_per_frame: np.ndarray,
    zero_frames: int,
    *,
    min_frames: Optional[int] = None,
    min_hits: Optional[int] = None,
) -> list[tuple[int, int]]:
    assert hits_per_frame.ndim == 1
    iszero = np.concatenate((
        [0, 1], # handle non-zero elements at the start 
        np.equal(hits_per_frame, 0).view(np.int8),
        [1, 0], # handle non-zero elements at the end
    )) # type: ignore
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2) - 1
    # discard zero ranges shorter than zero_frames
    ranges_keep = np.diff(ranges, axis=1).flatten() >= zero_frames
    ranges_keep[0] = ranges_keep[-1] = True
    ranges = ranges[ranges_keep]
    # convert zero ranges to non-zero ranges
    ranges = ranges.flat[1:-1].reshape(-1, 2)
    # require minimum number of frames
    if min_frames is not None:
        ranges = filter(
            lambda s: (s[1] - s[0]) > min_frames,
            ranges,
        )
    # require minimum number of hits
    if min_hits is not None:
        ranges = filter(
            lambda s: np.sum(hits_per_frame[s[0]:s[1]]) > min_hits,
            ranges,
        )
    return list(ranges)

################################################################################

hits_per_frame = hits_per_frame_nonsaturated(hits, 0.5)
segments = find_spills(
    np.where(hits_per_frame > 2000, 100, 0),
    len(hits_per_frame) // 3000,
    min_frames=400,
    min_hits=50,
)

total_hits = []
total_us = []
total_rate = []

for start, end in segments:
    segment_us = (end - start) * 20.
    segment_hits = np.sum(hits_per_frame[start:end])
    total_hits.append(segment_hits)
    total_us.append(segment_us)
    total_rate.append(segment_hits / segment_us)

# beam intensities, taken from PointCalc.xlsx
if ion == 'proton':
    intensities = np.array([
        2.00E+08,
        4.00E+08,
        8.00E+08,
        1.20E+09,
        2.00E+09,
        3.20E+09,
    ])
elif ion == 'carbon':
    # C12
    intensities = np.array([
        5.00E+06,
        1.00E+07,
        2.00E+07,
        3.00E+07,
        5.00E+07,
        8.00E+07,
    ])
else:
    raise ValueError()

################################################################################
# GROUP

def group_spills(counts: Iterable[int]) -> list[int]:
    cnt_iter = iter(counts)
    cnt_last = next(cnt_iter)
    intensity_indices = [0]
    intensity_index = 0
    for count in cnt_iter:
        if (count / cnt_last) > 1.15:
            intensity_index += 1
            cnt_last = count
        intensity_indices.append(intensity_index)
    return intensity_indices

int_indices = group_spills(total_rate)

print([[i] + list(x) for x, i in zip(segments, int_indices)])

if plot_type == 'curve':
    # save memory
    del hits
    # get timestamps
    # crop data
    prange = slice(None)
    #     np.argmax(times > times[segments[0][0]] - 3.0),
    #     np.argmax(times > times[segments[-1][1]] + 3.0),
    # )
    # times = times - times[prange.start]

    # times_int = [0 for _ in intensities]
    # for idx, (start, end) in zip(int_indices, segments):
    #     times_int[idx] += times[end] - times[start]

    # print(times_int)

    # plot curves
    plt.plot(hits_per_frame[prange])
    # plt.plot(bn.move_mean(hits_per_frame[prange], window=301, min_count=1))
    # plot segments
    for (start, end), int_idx in zip(segments, int_indices):
        plt.axvspan(start, end, color=f'C{int_idx}', alpha=0.5)
    # finish plot
    plt.ylabel('Hits/Frame')
    plt.xlabel('Time (s)')
    plot()
    exit()

################################################################################
# analyze pixels

sensor_size = hits.shape[1:]
# hitmap for each segment
segment_pixels = np.array([
    np.sum(hits[start:end, ...], axis=0)
    for start, end in segments
])
del hits

# hitmaps / microseconds per intensity
pixel_hits = np.zeros((1 + len(intensities), *sensor_size))
total_us_sum = np.zeros((1 + len(intensities),))
total_us_sum[0] = 1

for hits, us, idx in zip(segment_pixels, total_us, int_indices):
    pixel_hits[idx + 1] += hits
    total_us_sum[idx + 1] += us
del segment_pixels

# convert to rate
pixel_rate = pixel_hits / (1e-6 * total_us_sum[:,np.newaxis, np.newaxis])

################################################################################
# polyfit

fitdata = np.polynomial.polynomial.polyfit(
    x = np.concatenate(([0], intensities)), # type: ignore
    y = pixel_rate.reshape(len(pixel_rate), -1),
    deg = 2,
)
fitdata = fitdata.reshape(-1, *sensor_size)

plt.plot(fitdata[1].flat, fitdata[2].flat, 'x')
plt.show()

for c, i in zip(total_hits, intensities[int_indices]):
    print(c, i, c/i)



# plt.plot(intensities, total_hits)

# plt.show()
