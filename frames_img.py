#!/usr/bin/env python3.10
import argparse
import dataclasses
from pathlib import Path
from typing import Any, Optional

import h5py
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np

import frames.io
import util.gridscan

################################################################################

parser = argparse.ArgumentParser()

a_input_file = parser.add_argument(
    'input_file',
    help='h5 input file',
)

parser.add_argument(
    '--plot',
    choices=['hitmap', 'histogram'],
)

parser.add_argument(
    '--scan',
    metavar='A,B,C',
    help='scan indecies',
)

parser.add_argument(
    '--output',
    metavar='FILE',
    help='output file, show plot when not specified',
)

parser.add_argument(
    '--plot_range',
    type=float,
)

parser.add_argument(
    '--plot_range_min',
    type=float,
)

parser.add_argument(
    '--print_config',
    action='store_true',
)

parser.add_argument(
    '--logx',
    action='store_true',
)
parser.add_argument(
    '--logy',
    action='store_true',
)

parser.add_argument(
    '--total',
    action='store_true',
)

parser.add_argument(
    '--label',
)

parser.add_argument(
    '--title',
)

parser.add_argument(
    '--frames',
)


try:
    import argcomplete
    from argcomplete.completers import FilesCompleter
    setattr(a_input_file, 'completer', FilesCompleter('h5'))
    argcomplete.autocomplete(parser)
except ImportError:
    pass

################################################################################

args = parser.parse_args()

input_file: Path = Path(args.input_file)
plot_type: Optional[str] = args.plot
if args.scan:
    scan_idx: tuple[int, ...] = \
        tuple(int(i) for i in args.scan.split(','))
else:
    scan_idx = ()
output: Optional[Path] = args.output and Path(args.output)
print_config: bool = args.print_config
logx: bool = args.logx
logy: bool = args.logy
total: bool = args.total
plot_range: Optional[float] = args.plot_range
plot_range_min: Optional[float] = args.plot_range_min
label: Optional[str] = args.label
title: Optional[str] = args.title
frames_range: Optional[str] = args.frames

################################################################################

assert input_file.is_file(), 'input file does not exist'
if output and not plot_type:
    print('plot type must be specified when using --output')
    exit(1)

with h5py.File(args.input_file) as file:
    # get information about parameter scan
    if 'scan' in file:
        group_scan = file['scan']
        assert isinstance(group_scan, h5py.Group)
        scan_parameters, scan_shape = util.gridscan.load_scan(group_scan)
    else:
        scan_parameters, scan_shape = [], ()

    if len(scan_idx) != len(scan_shape):
        print('invalid --scan parameter, available parameters:')
        for p in scan_parameters:
            print(f'- {p.name:>12} ({len(p.values)} values)')
        exit(1)


    scan_idx_str = ''.join(f'_{i}' for i in scan_idx)
    group_name = 'frames' + scan_idx_str

    # old data format used 'scurve' prefix
    if not group_name in file:
        group_name = 'scurve' + scan_idx_str

    # load data
    group_frames = file[group_name]
    assert isinstance(group_frames, h5py.Group)
    config, hit_frames, _, _ = frames.io.load_frames(group_frames)

    # infinite runs
    if config.num_frames == 0:
        config.num_frames = hit_frames.shape[0]

if frames_range is not None:
    start, end = map(int, frames_range.split(':'))
    hit_frames = hit_frames[start:end]

hit_frames = np.sum(hit_frames, axis=0)
hit_frames = np.flip(hit_frames, axis=1)
if not total:
    hit_frames = hit_frames / (config.frame_length_us * config.num_frames * 1e-6)

################################################################################

config_dict = dataclasses.asdict(config)
del config_dict['dac_cfg']
for name, value in dataclasses.asdict(config.dac_cfg).items():
    config_dict['dac.'+name] = value

if print_config:
    for name, value in config_dict.items():
        print(f'{name:>20} = {value}')

if label:
    fmt_dict = dataclasses.asdict(config)
    del fmt_dict['dac_cfg']
    fmt_dict['dac'] = config.dac_cfg
    # label = label.format(**fmt_dict)
    label = str(eval(f'f"{label}"', fmt_dict))
    label = label.encode('latin-1','backslashreplace').decode('unicode_escape')

if title:
    fmt_dict = dataclasses.asdict(config)
    del fmt_dict['dac_cfg']
    fmt_dict['dac'] = config.dac_cfg
    # label = label.format(**fmt_dict)
    title = str(eval(f'f"{title}"', fmt_dict))
    title = title.encode('latin-1','backslashreplace').decode('unicode_escape')

################################################################################

hits_min = plot_range_min or 0
hits_max = plot_range or np.max(hit_frames)

if plot_type == 'hitmap':
    fig = plt.gcf()
    ax = fig.gca()

    if logy:
        norm = matplotlib.colors.LogNorm(1, hits_max)
        im_hits = plt.imshow(
            hit_frames,
            norm=norm,
        )
    else:
        im_hits = plt.imshow(
            hit_frames,
            vmin=hits_min,
            vmax=hits_max,
        )

    ax.set_ylabel('Pixel Row')
    ax.set_xlabel('Pixel Column')

    if label:
        plt.text(
            x=0.01, y=-0.12,
            s=label,
            fontsize=14, ha="left", va='top',
            transform=ax.transAxes,
        )
    if title:
        fig.suptitle(title)

    fig.colorbar(im_hits, ax=ax)
    fig.tight_layout()
    ax.set_title('Hits' if total else 'Hits/s')
elif plot_type == 'histogram':
    raise NotImplementedError()
else:
    print('no plot type specified')
    exit(0)

################################################################################

if output:
    for key in config_dict:
        config_dict[key] = str(config_dict[key])
    plt.savefig(output, dpi=300, bbox_inches='tight', metadata=config_dict)
else:
    plt.show()
