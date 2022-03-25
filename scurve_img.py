#!/usr/bin/env python3.10
import argparse
import dataclasses
from multiprocessing.sharedctypes import Value
from pathlib import Path
from typing import Any, Optional

import h5py
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np

import scurve.io
import scurve.analysis
import util.gridscan

################################################################################

parser = argparse.ArgumentParser()

a_input_file = parser.add_argument(
    'input_file',
    help='h5 input file',
)

parser.add_argument(
    '--plot',
    choices=['threshold', 'noise'],
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
    type=float, nargs=2,
)

parser.add_argument(
    '--print_config',
    action='store_true',
)

parser.add_argument(
    '--label',
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
plot_range: Optional[tuple[float, float]] = args.plot_range
label: Optional[str] = args.label

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
    group_name = 'scurve' + scan_idx_str

    # load data
    group_frames = file[group_name]
    assert isinstance(group_frames, h5py.Group)
    config, hits_signal, hits_noise = scurve.io.load_scurve(group_frames)


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
    label = str(eval(f'f"{label}"', fmt_dict))
    label = label.encode('latin-1','backslashreplace').decode('unicode_escape')

################################################################################

if not plot_type:
    print('no plot type specified')
    exit(0)

assert hits_noise is not None
hits_noise = np.flip(hits_noise, axis=-1)
hits_signal = np.flip(hits_signal, axis=-1)

sensor_size = hits_signal.shape[1:]
pixel_pos = np.meshgrid(*(np.array(np.arange(size)) for size in sensor_size))

################################################################################

# result arrays
efficiency = hits_signal / config.injections_total

threshold, noise = scurve.analysis.fit_sigmoids(
    config.injection_voltage,
    efficiency,
)
# convert noise to mV
noise *= 1e3

################################################################################


fig = plt.gcf()
ax = fig.gca()

if plot_type == 'threshold':
    data = threshold
    ax.set_title('Threshold (V)')
elif plot_type == 'noise':
    data = noise
    ax.set_title('Noise (mV)')
else:
    raise ValueError()

if plot_range is None:
    plot_range = np.min(data), np.max(data)

img = plt.imshow(
    data,
    vmin=plot_range[0],
    vmax=plot_range[1],
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

fig.colorbar(img, ax=ax)
fig.tight_layout()

################################################################################

if output:
    for key in config_dict:
        config_dict[key] = str(config_dict[key])
    plt.savefig(output, dpi=300, bbox_inches='tight', metadata=config_dict)
else:
    plt.show()
