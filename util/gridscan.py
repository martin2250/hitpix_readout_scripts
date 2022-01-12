from typing import Iterable
import h5py
from dataclasses import dataclass
import numpy as np


@dataclass
class ParameterScan:
    name: str
    values: np.ndarray

def load_scan(group_scan: h5py.Group) -> tuple[list[ParameterScan], tuple[int, ...]]:
    scan_parameters: list[ParameterScan] = []
    # extract names
    assert 'scan_names' in group_scan.attrs, 'scan group incomplete'
    scan_names = group_scan.attrs['scan_names']
    assert isinstance(scan_names, np.ndarray)
    # check if old method was used to encode scan values
    if 'scan_values' in group_scan.attrs:
        # old method, all scans must have the same number of values
        scan_values = group_scan.attrs['scan_values']
        assert isinstance(scan_values, np.ndarray)
        assert scan_values.ndim == 2
        for name, values in zip(scan_names, scan_values):
            assert isinstance(name, str)
            assert isinstance(values, np.ndarray)
            scan_parameters.append(ParameterScan(name, values))
    else:
        # new method, every parameter scan is saved as an individual dataset
        for name in scan_names:
            assert isinstance(name, str)
            assert name in group_scan
            dset_values = group_scan[name]
            assert isinstance(dset_values, h5py.Dataset)
            values = dset_values[:]
            assert isinstance(values, np.ndarray)
            assert values.ndim == 1
            scan_parameters.append(ParameterScan(name, values))
    # return parameters and shape of grid scan
    scan_shape = tuple(len(param.values) for param in scan_parameters)
    return scan_parameters, scan_shape

def save_scan(group_scan: h5py.Group, scan_parameters: list[ParameterScan]) -> None:
    '''use with require_group'''
    # check if group already contains a scan
    if 'scan_names' in group_scan.attrs:
        # group already contains scan, ensure that parameters are equal
        scan_parameters_f, _ = load_scan(group_scan)
        assert len(scan_parameters) == len(scan_parameters_f), 'existing scan has different number of parameters'
        for a, b in zip(scan_parameters, scan_parameters_f):
            assert a.name == b.name, 'existing scan has different parameters'
            assert np.array_equal(a.values, b.values), f'existing scan has different values for parameter {a.name}'
    else:
        # group is 'fresh', save scan parameters
        group_scan.attrs['scan_names'] = [param.name for param in scan_parameters]
        for param in scan_parameters:
            group_scan.create_dataset(param.name, data=param.values)

def parse_range(range_str: str) -> np.ndarray:
    if range_str.startswith('['):
        assert range_str.endswith(']')
        parts = range_str[1:-1].split(',')
        return np.array(float(p) for p in parts)
    else:
        parts = range_str.split(':')
        assert len(parts) in (3, 4)
        start, stop = map(float, parts[:2])
        count = int(parts[2])
        scale = parts[3] if len(parts) == 4 else 'lin'
        if scale in ('lin', 'linear'):
            return np.linspace(start, stop, count)
        elif scale in ('log', 'logarithmic'):
            return np.geomspace(start, stop, count)
        else:
            raise ValueError(f'{scale} is not a proper scale type')


def parse_scan(args_scan: list[str]) -> tuple[list[ParameterScan], tuple[int, ...]]:
    scan_parameters: list[ParameterScan] = []
    # loop over all parameters
    for scan in args_scan:
        name, scan_range = map(str.strip, scan.split('='))
        values = parse_range(scan_range)
        scan_parameters.append(ParameterScan(name, values))
    # return parameters and shape of grid scan
    scan_shape = tuple(param.values.size for param in scan_parameters)
    return scan_parameters, scan_shape

def apply_scan(param_dict: dict, scan_parameters: list[ParameterScan], idx: tuple[int, ...]) -> None:
    assert len(idx) == len(scan_parameters)
    for i, param in zip(idx, scan_parameters):
        # match type if parameter already exists, otherwise use parameter as float
        try:
            ptype = eval(f'type({param.name})', param_dict)
        except NameError:
            ptype = float
        # convert value to proper type
        value = ptype(param.values[i])
        # assign value
        exec(f'{param.name} = {value}', param_dict)

def apply_set(param_dict: dict, args_set: list[str]) -> None:
    for p_set in args_set:
        name, _, value = map(str.strip, p_set.partition('='))
        try:
            exec(f'{name} = type({name})({value})', param_dict)
        except NameError:
            exec(f'{name} = {value}', param_dict)

def group_name(prefix: str, scan_idx: Iterable[int]) -> str:
    return prefix + ''.join(f'_{i}' for i in scan_idx)
