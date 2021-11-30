#!/usr/bin/python
import dataclasses
import datetime
import h5py
import time
from dataclasses import dataclass
import argparse
import scipy.optimize
from typing import cast, Any, Union
import matplotlib.pyplot as plt
import scurves
import matplotlib.axes

import numpy as np

################################################################################
# sigmoid curve fitting

def fitfunc_sigmoid(x, threshold, width):
    e = (x - threshold) / width
    return 1 / (1 + np.exp(-e))

def fit_sigmoid(injection_voltage: Union[np.ndarray, list], efficiency: np.ndarray) -> tuple[float, float]:
    assert len(efficiency.shape) == 1
    (threshold, width), _ = cast(
        tuple[np.ndarray, Any],
        scipy.optimize.curve_fit(
            fitfunc_sigmoid,
            injection_voltage,
            efficiency,
            bounds=[(0, 0), (2, 1)]
        ),
    )
    return threshold, width

def fit_sigmoids(injection_voltage: Union[np.ndarray, list], efficiency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shape_output = efficiency.shape[1:]
    threshold, width = np.zeros(shape_output), np.zeros(shape_output)
    for idx in np.ndindex(*shape_output):
            efficiency_pixel = efficiency[:,idx]
            t, w = fit_sigmoid(injection_voltage, efficiency_pixel)
            threshold[idx] = t
            width[idx] = w
    return threshold, width

################################################################################

def plot_single_scurve(injection_voltage: Union[np.ndarray, list], efficiency: np.ndarray, ax: matplotlib.axes.Axes):
    popt = fit_sigmoid(injection_voltage, efficiency)
    
    x_fit = np.linspace(np.min(injection_voltage), np.max(injection_voltage), 200)
    y_fit = fitfunc_sigmoid(x_fit, *popt)

    ax.plot(injection_voltage, efficiency, 'x')
    ax.plot(x_fit, y_fit, 'r')

################################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'input_file',
        help='h5 input file',
    )

    parser.add_argument(
        '--h5group',
        default='auto',
        help='h5 group name',
    )

    parser.add_argument(
        '--show_config',
        action='store_true',
        help='show scurve config',
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
        h5group = args.h5group
        if h5group not in file:
            if h5group == 'auto':
                group_names = [name for name, item in file.items() if isinstance(item, h5py.Group)]
                h5group = group_names[0]
                if len(group_names) > 1:
                    print(f'{len(group_names)} groups in {args.input_file}, using first group {h5group}')
            else:
                print(f'group {h5group} not found in {args.input_file}')
        # load group
        group = file[h5group]
        assert isinstance(group, h5py.Group)
        config, hits_signal, hits_noise = scurves.load_scurve(group)
    
    ################################################################################

    if args.single_scurve:
        x, y = args.single_scurve
        efficiency_pixel = hits_signal[:,x,y] / config.injections_total
        fig, ax = plt.subplots()
        plot_single_scurve(config.injection_voltage, efficiency_pixel, ax)
        plt.show()

    ################################################################################

    if args.maps:
        sensor_size = hits_noise.shape[1:]
        map_types = args.maps
        if 'all' in map_types:
            map_types = map_choices[1:]
        pixel_edges = tuple(np.arange(size + 1) - 0.5 for size in sensor_size)
        pixel_x, pixel_y = np.meshgrid(*(np.array(np.arange(size)) for size in sensor_size))
        # calculate pixel properties
        threshold, width = fit_sigmoids(config.injection_voltage, hits_signal / config.injections_total)
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