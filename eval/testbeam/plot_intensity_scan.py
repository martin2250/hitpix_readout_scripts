#!/usr/bin/env python
'''
'''

import argparse
import collections
from dataclasses import dataclass, field
import json
from typing import Optional, cast
from matplotlib.image import AxesImage

import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
import itertools
import scipy.optimize

if True:  # do not reorder with autopep8 or sortimports
    sys.path.insert(1, str(Path(__file__).parents[2]))
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
parser.add_argument('--stack', choices=('horizontal', 'vertical'))
parser.add_argument('--title', action='append')
parser.add_argument('--mask', action='append')
parser.add_argument('--suptitle')
parser.add_argument('--fit_gauss', action='store_true')
parser.add_argument('--share_axes', action='store_true')
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
titles: list[str] = args.title or []
extra_masks: list[str] = args.mask or []
plot_type: str = args.plot_type
save: Optional[Path] = args.save
stack: Optional[str] = args.stack
suptitle: Optional[str] = args.suptitle
mapping_path: Path = args.mapping_file
fit_gauss: bool = args.fit_gauss
share_axes: bool = args.share_axes

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
    fig, axes = plt.subplots(len(input_files), 1, sharex=True, sharey=share_axes) # nopep8
    axes = cast(list[plt.Axes], axes) # not correct but good enough for pylance
    axes_x = axes[-1:]
    axes_y = axes
    if len(input_files) > 2:
        idx = len(input_files) // 2
        axes_y = axes[idx:idx+1]
elif stack == 'horizontal':
    fig, axes = plt.subplots(1, len(input_files), sharey=True, sharex=share_axes) # nopep8
    axes = cast(list[plt.Axes], axes) # not correct but good enough for pylance
    axes_x = axes
    axes_y = axes[:1]
else:
    fig = plt.gcf()
    axes = [plt.gca()] * len(input_files)
    axes_x = axes_y = axes[:1]

################################################################################

images: list[AxesImage] = []
linestyles = itertools.cycle(['solid', 'dotted', 'dashed', 'dashdot'])

for input_file, ax, title, linestyle in zip(input_files, axes, titles, linestyles):
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
        hits = dset.astype(np.uint16)[()] # 16 bits is enough for frames and adders
        assert isinstance(hits, np.ndarray)
        # flip hitmap along x
        hits = np.flip(hits, axis=-1)
        # account for overflowing counters
        if mapping.adders:
            # subtract offset (faulty daq calculation)
            hits_sub = np.where(hits >= 48, hits - 48, 0)
            hits = hits_sub // 255 # how many pixels had any hits at all?
            hits_sub -= hits * 255 # subtract overflowing counters
            # how many additional hits were registered?
            hits += np.where(hits_sub > 0, 255 - hits_sub, 0)
        # apply mask
        for mask in itertools.chain(mapping.mask, extra_masks):
            mask = parse_ndrange(mask, 1 if mapping.adders else 2)
            hits[(...,) + mask] = 0
        # reshape hits to make adders also 2D
        if mapping.adders:
            hits = hits.reshape(hits.shape[0], 1, hits.shape[-1])
        # cleanup
        del dset_times, dset, group_frames

    ############################################################################

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
    
    ############################################################################
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

    ############################################################################

    # use weakest intensity only for fit
    hitmap_fit = intensity_hits[0] / intensity_us[0]
    intensity_factor = 1.0 # factor between sensor area and full
    beam_profile: Optional[np.ndarray] = None

    if fit_gauss:
        if mapping.adders:
            def gauss_1d(X, posx: float, total_int: float, sigma: float):
                return total_int / (np.sqrt(2 * np.pi) * sigma) * np.exp(
                    -np.square(X - posx) / (2 * np.square(sigma))
                )
            X_fit = np.arange(sensor_size[-1])
            popt, pcov = scipy.optimize.curve_fit(
                gauss_1d,
                X_fit, hitmap_fit.flatten(),
                p0 = (sensor_size[-1] / 2, 1000, 5),
            )
            posx, total_int, sigma = popt
            intensity_factor = total_int / np.sum(hitmap_fit)
            beam_profile = gauss_1d(X_fit, *popt).reshape(sensor_size)
        else:
            def gauss_2d(X, posx: float, posy: float, total_int: float, sigma: float):
                return total_int / (2 * np.pi * sigma * sigma) * np.exp(
                    -(
                        np.square(X[0] - posx) + np.square(X[1] - posy)
                    ) / (2 * np.square(sigma))
                )
            X_fit = np.indices(sensor_size)
            popt, pcov = scipy.optimize.curve_fit(
                gauss_2d,
                X_fit.reshape(2, -1), hitmap_fit.flatten(),
                p0 = tuple(x / 2 for x in sensor_size) + (1000, 5),
            )
            posx, posy, total_int, sigma = popt
            intensity_factor = total_int / np.sum(hitmap_fit)
            beam_profile = gauss_2d(X_fit, *popt)

    ############################################################################


    if plot_type == 'hitmap':
        hitmap = np.sum(intensity_hits, axis=0) / np.sum(intensity_us)
        im = ax.imshow(hitmap)
        images.append(im)
        fig.colorbar(im, ax=ax, label='Hits/µs')
        continue
    if plot_type == 'pixel_intensity':
        if fit_gauss:
            # group pixels by beam intensity and plot each group as a line
            assert beam_profile is not None
            num_groups = 5
            pixel_intensity_index = np.ceil(num_groups * beam_profile / np.max(beam_profile)) - 1
            for igroup in range(num_groups):
                pixel_in_group = pixel_intensity_index == igroup
                intensity_group_hits_us = np.sum(intensity_hits_us[..., pixel_in_group], axis=1) # nopep8
                intensity_group_hits_us /= np.sum(pixel_in_group)
                ax.plot(
                    intensity_ion_us,
                    intensity_group_hits_us,
                    'x',
                    color=f'C{igroup}',
                    ls=linestyle if stack is None else 'solid',
                    label=title if igroup == 0 else None,
                )
        else:
            # plot each pixel as a line
            ax.plot(
                intensity_ion_us,
                intensity_hits_us.reshape(len(intensity_hits_us), -1),
                'x-', alpha=0.3
            )
        continue
    if plot_type == 'total_intensity':
        ax.plot(
            intensity_ion_us,
            intensity_factor * np.sum(intensity_hits_us, axis=(1, 2)),
            'x-', label=title,
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
        ax.plot(1000 * fitdata[1], 1000 * fitdata[2] / fitdata[1], 'x', label=title)

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
    if share_axes:
        vmax = max(im.get_clim()[1] for im in images)
        for im in images:
            im.set_clim(0, vmax)
elif plot_type == 'pixel_intensity':
    for ax in axes_x: ax.set_xlabel('Beam Intensity (1/us)')
    if stack is None:
        plt.legend()
    if fit_gauss:
        for ax in axes_y: ax.set_ylabel('Mean Pixel Hit Rate (1/µs)')
    else:
        for ax in axes_y: ax.set_ylabel('Pixel Hit Rate (1/µs)')
    for ax in axes:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
elif plot_type == 'total_intensity':
    for ax in axes_x: ax.set_xlabel('Beam Intensity (1/us)')
    if fit_gauss:
        for ax in axes_y: ax.set_ylabel('Hit Rate, adjusted to full Beam (1/µs)')
    else:
        for ax in axes_y: ax.set_ylabel('Sensor Hit Rate (1/µs)')
    if stack is None:
        plt.legend()
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
