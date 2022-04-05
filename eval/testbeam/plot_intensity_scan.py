#!/usr/bin/env python
'''
'''

import argparse
from dataclasses import dataclass, field
import json
from optparse import Option
from typing import Iterable, Optional, cast
from matplotlib.image import AxesImage

import h5py
import matplotlib.pyplot as plt
import numpy as np
from rich import print
import sys
from pathlib import Path
import bottleneck as bn
import itertools

if True:  # do not reorder with autopep8 or sortimports
    sys.path.insert(1, str(Path(__file__).parents[2]))
import frames.io
from util.time_sync import TimeSync
from util import parse_ndrange

################################################################################

parser = argparse.ArgumentParser()

a_input_file = parser.add_argument(
    'input_file', nargs='+', type=Path,
    help='h5 input file',
)
parser.add_argument(
    '--plot_type', required=True,
    choices=('curve', 'total_intensity', 'pixel_intensity', 'fit_scatter', 'hitmap'),
)
parser.add_argument(
    '--stack',
    choices=('horizontal', 'vertical'),
)
parser.add_argument(
    '--title',
    action='append',
)
parser.add_argument(
    '--suptitle',
)
parser.add_argument(
    '--mapping_file',
    type=Path, default=Path('/home/martin/Desktop/master-thesis/2022-03-27-testbeam/intensity_mapping.json'),  # nopep8
)
parser.add_argument('--save', type=Path)

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

input_files: list[Path] = args.input_file
titles: list[str] = args.title
plot_type: str = args.plot_type
save: Optional[Path] = args.save
stack: Optional[str] = args.stack
suptitle: Optional[str] = args.suptitle
mapping_path: Path = args.mapping_file

del args

titles += [f.name for f in input_files[len(titles):]]

################################################################################


@dataclass
class Mapping:
    ion: str
    energy: int
    chip: int
    adders: bool
    intensitites: list[int]
    segments: list[tuple[int, int, int]]
    mask: list[str] = field(default_factory=list)


with open(mapping_path) as fm:
    mapping_file = {
        key: Mapping(**value)
        for key, value in json.load(fm).items()
    }

################################################################################

# I<x> = intensity_table[x] ions/s
intensity_table = {
    'proton': np.array([0, 8.00E+07, 1.20E+08, 2.00E+08, 3.20E+08, 4.00E+08, 6.00E+08, 8.00E+08, 1.20E+09, 2.00E+09, 3.20E+09]),  # nopep8
    'carbon': np.array([0, 2.00E+06, 3.00E+06, 5.00E+06, 8.00E+06, 1.00E+07, 1.50E+07, 2.00E+07, 3.00E+07, 5.00E+07, 8.00E+07]),  # nopep8
}

################################################################################

if stack == 'vertical':
    fig, axes = plt.subplots(len(input_files), 1, sharex=True, sharey=True)
    axes = cast(list[plt.Axes], axes) # not correct but good enough for pylance
    axes_x = axes[-1:]
    axes_y = axes
elif stack == 'horizontal':
    fig, axes = plt.subplots(1, len(input_files), sharey=True, sharex=True)
    axes = cast(list[plt.Axes], axes) # not correct but good enough for pylance
    axes_x = axes
    axes_y = axes[:1]
else:
    fig = plt.gcf()
    axes = [plt.gca()] * len(input_files)
    axes_x = axes_y = axes[:1]

################################################################################

images: list[AxesImage] = []

for input_file, ax, title in zip(input_files, axes, titles):
    if stack is not None:
        ax.set_title(title)
    mapping = mapping_file[input_file.name]
    # load file data
    with h5py.File(input_file) as file:
        # load frames group
        group_frames = file['frames']
        assert isinstance(group_frames, h5py.Group)
        # load times dataset
        dset_times = group_frames['times']
        assert isinstance(dset_times, h5py.Dataset)
        times_raw = dset_times[()]
        assert isinstance(times_raw, np.ndarray)
        sync_dict = cast(str, dset_times.attrs['sync'])
        time_sync = TimeSync.fromdict(json.loads(sync_dict))
        # load frames dataset
        config_dict = json.loads(cast(str, group_frames.attrs['config']))
        frame_us = cast(float, config_dict['frame_length_us'])
        # load actual data
        dset = group_frames['adders' if mapping.adders else 'frames']
        assert isinstance(dset, h5py.Dataset)
        hits = dset[()]
        assert isinstance(hits, np.ndarray)
        # apply mask
        for mask in mapping.mask:
            mask = parse_ndrange(mask, 1 if mapping.adders else 2)
            hits[..., mask] = 0
        # reshape hits to make adders also 2D
        if mapping.adders:
            hits = hits[..., np.newaxis]
            hits = np.max((hits - hits[0])[np.newaxis, ...], 0)
        # cleanup
        del dset_times, dset, group_frames

    # plot hit rate over time
    if plot_type == 'curve':
        # we don't need the absolute time, improve precision
        time_sync.sync_time = 0
        times_sec = time_sync.convert(times_raw)
        # time relative to first spill
        times_sec -= times_sec[mapping.segments[0][1]]
        # plot 3s before and after all spills
        prange = slice(
            np.argmax(times_sec > (times_sec[mapping.segments[0][1]] - 3.0)),
            np.argmax(times_sec > (times_sec[mapping.segments[-1][2]] + 3.0)),
        )
        ax.plot(
            times_sec[prange],
            np.sum(hits[prange, ...], axis=(1, 2)) / frame_us,
            label=title,
        )
        del hits
        if len(input_files) == 1 or stack is not None:
            for idx, start, end in mapping.segments:
                ax.axvspan(
                    times_sec[start], times_sec[end],
                    color=f'C{idx}', alpha=0.5,
                )
        continue

    # sum up spills of same intensity
    sensor_size = hits.shape[1:]
    # hitmap for each intensity
    intensity_hits = np.zeros((len(mapping.intensitites), *sensor_size))
    intensity_us = np.ones((len(mapping.intensitites),))

    intensity_ion_us = intensity_table[mapping.ion][mapping.intensitites] * 1e-6

    for idx, start, end in mapping.segments:
        intensity_hits[idx] += np.sum(hits[start:end], axis=0)
        intensity_us[idx] += (end - start) * frame_us

    intensity_hits_us = \
        intensity_hits / intensity_us[..., np.newaxis, np.newaxis]

    if plot_type == 'hitmap':
        im = ax.imshow(np.sum(intensity_hits_us, axis=0))
        images.append(im)
        fig.colorbar(im, ax=ax, label='Hits/µs')
        continue
    if plot_type == 'pixel_intensity':
        ax.plot(
            intensity_ion_us,
            intensity_hits_us.reshape(len(intensity_hits_us), -1),
            'x-', alpha=0.3, label=title,
        )
        continue
    if plot_type == 'total_intensity':
        ax.plot(
            intensity_ion_us,
            np.sum(intensity_hits_us, axis=(1, 2)),
            'x', label=title,
        )
        continue

    # fit parabola to intensity
    fitdata_x = np.concatenate(([0], intensity_ion_us))
    fitdata_y = np.concatenate([
        np.zeros((1, np.prod(sensor_size))),
        intensity_hits_us.reshape(len(intensity_hits_us), -1),
    ])
    fitdata = np.polynomial.polynomial.polyfit(
        x=fitdata_x,
        y=fitdata_y,
        deg=2,
        w=[100] + [1] * len(intensity_hits_us),
    )
    if plot_type == 'fit_scatter':
        ax.plot(fitdata[1], fitdata[2] / fitdata[1], 'x', label=title)

################################################################################

if suptitle is not None:
    fig.suptitle(suptitle)

################################################################################

if plot_type == 'curve':
    if stack is None:
        plt.legend()
    for ax in axes_x: ax.set_xlabel('Time (s)')
    for ax in axes_y: ax.set_ylabel('Hit Rate (1/µs)')
elif plot_type == 'hitmap':
    for ax in axes_x: ax.set_xlabel('Pixel X')
    for ax in axes_y: ax.set_ylabel('Pixel Y')
    vmax = max(im.get_clim()[1] for im in images)
    for im in images:
        im.set_clim(0, vmax)
elif plot_type in ['total_intensity', 'pixel_intensity']:
    if stack is None:
        plt.legend()
    for ax in axes_x: ax.set_xlabel('Beam Intensity (1/us)')
    for ax in axes_y: ax.set_ylabel('Hit Rate (1/µs)')
    for ax in axes:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
elif plot_type == 'fit_scatter':
    if stack is None:
        plt.legend()
    for ax in axes_x: ax.set_xlabel('Linear Component')
    for ax in axes_y: ax.set_ylabel('Relative Quadratic Component')
else:
    raise ValueError(f'invalid plot_type "{plot_type}"')

################################################################################

if save is not None:
    plt.savefig(save, dpi=300, transparent=False, bbox_inches='tight')
else:
    plt.show()
