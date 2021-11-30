#!/usr/bin/python
import dataclasses
import datetime
import h5py
import time
from dataclasses import dataclass
import argparse
import scipy.optimize
from typing import cast, Any
import matplotlib.pyplot as plt

import numpy as np


################################################################################

parser = argparse.ArgumentParser()

parser.add_argument(
    'input_file',
    help='h5 input file',
)

parser.add_argument(
    '--single_scurve', nargs=2,
    metavar=('x', 'y'),
    type=int,
    help='plot scurve for single pixel',
)

map_choices = ('all', 'threshold', 'width', 'noise')
parser.add_argument(
    '--maps', nargs='+',
    choices=map_choices,
    help='plot map of thresholds',
)

args = parser.parse_args()

################################################################################
# load data
with h5py.File(args.input_file) as file:
    # load group
    group = file['injection_ramp']
    assert isinstance(group, h5py.Group)
    # load attributes
    measurement_time = group.attrs['measurement_time']
    injection_voltages = group.attrs['injection_voltages']
    injections = group.attrs['injections']
    injections_per_round = group.attrs['injections_per_round']
    sensor_size = cast(tuple[int, int], tuple(cast(np.ndarray, group.attrs['sensor_size']).flat))
    sendor_dac = group.attrs['sendor_dac']
    # load datasets
    dset_signal = group['hits_signal']
    dset_noise = group['hits_noise']
    assert isinstance(dset_signal, h5py.Dataset)
    assert isinstance(dset_noise, h5py.Dataset)
    hits_signal = dset_signal[()]
    hits_noise = dset_noise[()]
    assert isinstance(hits_signal, np.ndarray)
    assert isinstance(hits_noise, np.ndarray)
    

################################################################################
# sigmoid curve fitting

def fitfunc_sigmoid(x, threshold, width):
    e = (x - threshold) / width
    return 1 / (1 + np.exp(-e))

def fit_sigmoid(efficiency: np.ndarray) -> tuple[float, float]:
    (threshold, width), _ = cast(
        tuple[np.ndarray, Any],
        scipy.optimize.curve_fit(
            fitfunc_sigmoid,
            injection_voltages,
            efficiency_pixel,
            bounds=[(0, 0), (2, 1)]
        ),
    )
    return threshold, width


################################################################################

if args.single_scurve:
    x, y = args.single_scurve

    efficiency_pixel = hits_signal[:,x,y] / injections
    popt = fit_sigmoid(efficiency_pixel)
    
    x_fit = np.linspace(np.min(injection_voltages), np.max(injection_voltages), 200)
    y_fit = fitfunc_sigmoid(x_fit, *popt)

    fig, ax = plt.subplots()
    ax.plot(injection_voltages, efficiency_pixel, 'x')
    ax.plot(x_fit, y_fit, 'r')

    plt.show()

################################################################################


if args.maps:
    map_types = args.maps
    if 'all' in map_types:
        map_types = map_choices[1:]
    pixel_edges = tuple(np.arange(size + 1) - 0.5 for size in sensor_size)
    pixel_x, pixel_y = np.meshgrid(*(np.array(np.arange(size)) for size in sensor_size))
    # calculate pixel properties
    threshold, width = np.zeros(sensor_size), np.zeros(sensor_size)
    for x, y in np.ndindex(*sensor_size):
        efficiency_pixel = hits_signal[:,x,y] / injections
        t, w = fit_sigmoid(efficiency_pixel)
        threshold[x, y] = t
        width[x, y] = w
    # find broken pixels
    too_high = threshold > 1.5
    too_low = threshold < 0
    # mask arrays
    threshold = np.where(too_high | too_low, np.nan, threshold)
    width = np.where(too_high | too_low, np.nan, width)
    # plot
    fig, axes = plt.subplots(1, len(map_types), squeeze=False, sharey=True, figsize=(6*len(map_types), 5))
    for ax, map_type in zip(axes[0], map_types):
        # keep pcolormesh object for colorbar
        pcb = False
        # check what map to draw
        if map_type == 'threshold':
            ax.set_title('Threshold (V)')
            pcb = ax.pcolormesh(*pixel_edges, threshold)
        elif map_type == 'width':
            ax.set_title('Turn on Width(V)')
            pcb = ax.pcolormesh(*pixel_edges, width)
        elif map_type == 'noise':
            ax.set_title('Noise hits')
            pcb = ax.pcolormesh(*pixel_edges, np.sum(hits_noise, axis=0))
        if too_high.any():
            ax.plot(pixel_x[too_high], pixel_y[too_high], 'rx', label='dead')
        if too_low.any():
            ax.plot(pixel_x[too_low], pixel_y[too_low], 'rx', label='noisy')
        assert pcb
        ax.set_aspect('equal')
        plt.colorbar(pcb, ax=ax)
    plt.show()