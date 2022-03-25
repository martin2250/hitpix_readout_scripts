#!/usr/bin/env python
import argparse
import subprocess
from pathlib import Path

import h5py
from h5py._hl.group import Group

################################################################################


def walk_group_raw(group: h5py.Group, prefix: str = ''):
    for name, item in group.items():
        assert isinstance(item, h5py.Group) or isinstance(item, h5py.Dataset)

        path = prefix + '/' + name

        if isinstance(item, h5py.Dataset):
            print(
                f'{path} D {item.dtype}[{", ".join(str(i) for i in item.shape)}]')
        else:
            print(f'{path} G')

        for name, attr in item.attrs.items():
            print(f'{path}/{name} A  = {attr}')

        if isinstance(item, h5py.Group):
            walk_group_raw(item, prefix=path)

################################################################################

def parse_file(file: h5py.Group):
    pass

################################################################################


parser = argparse.ArgumentParser()

a_input_file = parser.add_argument(
    'input_file',
    help='h5 input file',
)

parser.add_argument('--walk', action='store_true')

try:
    import argcomplete
    from argcomplete.completers import FilesCompleter
    setattr(a_input_file, 'completer', FilesCompleter('h5'))
    argcomplete.autocomplete(parser)
except ImportError:
    pass

args = parser.parse_args()

################################################################################

with h5py.File(args.input_file) as file:
    if args.walk:
        walk_group_raw(file)
        exit(0)
    
    parent = Path(__file__).parent
    
    for name, item in file.items():
        name: str
        if name.startswith('scurve'):
            ret = subprocess.run([parent / 'scurve_plot.py', args.input_file])
            exit(ret.returncode)
        elif name.startswith('frames'):
            ret = subprocess.run([parent / 'frames_plot.py', args.input_file])
            exit(ret.returncode)
