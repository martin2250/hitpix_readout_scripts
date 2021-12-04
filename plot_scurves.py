#!/usr/bin/python
import dataclasses
import datetime
import typing
import h5py
import time
from dataclasses import dataclass
import argparse
import scipy.optimize
from typing import Literal, cast, Any, Union
import matplotlib.pyplot as plt
import daq.scurve
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent

import numpy as np

################################################################################
# sigmoid curve fitting

def fitfunc_sigmoid(x, threshold, noise):
    e = (x - threshold) / noise
    return 1 / (1 + np.exp(-e))

# sigmoid with inverse noise -> speed up fit
def fitfunc_sigmoid_inv(x, threshold, noise):
    e = (x - threshold) * noise
    return 1 / (1 + np.exp(-e))

def fit_sigmoid(injection_voltage: np.ndarray, efficiency: np.ndarray) -> tuple[float, float]:
    # check inputs
    assert len(efficiency.shape) == 1
    if injection_voltage.shape != efficiency.shape:
        print(injection_voltage.shape, efficiency.shape)
    assert injection_voltage.shape == efficiency.shape
    # check for dead / noisy pixels
    if sum(efficiency > 0.3) == 0: # dead
        return np.inf, np.nan
    if sum(efficiency < 0.7) == 0: # noisy
        return -np.inf, np.nan
    # get starting values
    i_threshold = np.argmax(efficiency > 0.5*(np.min(efficiency) + np.max(efficiency)))
    threshold_0 = injection_voltage[i_threshold]
    # fit
    (threshold, noise), _ = cast(
        tuple[np.ndarray, Any],
        scipy.optimize.curve_fit(
            fitfunc_sigmoid_inv,
            injection_voltage,
            efficiency,
            sigma=2-efficiency*(1-efficiency),
            p0=(threshold_0, 100),
            xtol=1e-3,
            gtol=1e-4,
        ),
    )
    return threshold, 1/noise

def fit_sigmoids(injection_voltage: np.ndarray, efficiency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shape_output = efficiency.shape[1:]
    threshold, noise = np.zeros(shape_output), np.zeros(shape_output)
    for idx in np.ndindex(*shape_output):
        efficiency_pixel = efficiency[(..., *idx)]
        try:
            t, w = fit_sigmoid(injection_voltage, efficiency_pixel)
            threshold[idx] = t
            noise[idx] = w
        except RuntimeError:
            threshold[idx] = noise[idx] = np.nan
    return threshold, noise

################################################################################

def plot_single_scurve(ax: Axes, injection_voltage: np.ndarray, efficiency: np.ndarray):
    popt = fit_sigmoid(injection_voltage, efficiency)
    
    x_fit = np.linspace(np.min(injection_voltage), np.max(injection_voltage), 200)
    y_fit = fitfunc_sigmoid(x_fit, *popt)

    ax.plot(injection_voltage, efficiency, 'x')
    ax.plot(x_fit, y_fit, 'r')

def plot_single_map(ax: Axes, threshold: np.ndarray, noise: np.ndarray, pixel_edges: tuple[np.ndarray, np.ndarray], pixel_pos: tuple[np.ndarray, np.ndarray], map_type: Literal['threshold', 'noise']):
    # find broken pixels
    too_high = threshold > 1.5
    too_low = threshold < 0
    # mask arrays
    threshold = np.where(too_high | too_low, np.nan, threshold)
    noise = np.where(too_high | too_low, np.nan, noise)
    # PLOT
    # keep pcolormesh object for colorbar
    pcb = False
    # check what map to draw
    if map_type == 'threshold':
        ax.set_title('Threshold (V)')
        pcb = ax.pcolormesh(*pixel_edges, threshold.T)
    elif map_type == 'noise':
        ax.set_title('Noise (V)')
        pcb = ax.pcolormesh(*pixel_edges, noise.T)
    else:
        raise ValueError(f'unknown map_type {map_type}')
    # draw dead / noisy pixels
    pixel_x, pixel_y = pixel_pos
    if too_high.any():
        ax.plot(pixel_x[too_high], pixel_y[too_high], 'rx', label='dead')
    if too_low.any():
        ax.plot(pixel_x[too_low], pixel_y[too_low], 'rx', label='noisy')
    ax.set_aspect('equal')
    plt.colorbar(pcb, ax=ax)

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
        '--pixel', nargs=2,
        metavar=('x', 'y'),
        type=int,
        help='plot scurve for single pixel',
    )

    map_choices = ('all', 'threshold', 'noise')
    parser.add_argument(
        '--maps', nargs='+',
        choices=map_choices,
        help='plot map of thresholds',
    )

    parser.add_argument(
        '--map_interactive',
        action='store_true',
        help='plot maps and show single scurves',
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
        config, hits_signal, hits_noise = daq.scurve.load_scurve(group)
    
    sensor_size = hits_noise.shape[1:]
    # pixel edges for pcolormesh
    pixel_edges = cast(tuple[np.ndarray, np.ndarray], tuple(np.arange(size + 1) - 0.5 for size in sensor_size))
    # pixel indices
    pixel_pos = np.meshgrid(*(np.array(np.arange(size)) for size in sensor_size))

    # calculate pixel properties
    efficiency = hits_signal / config.injections_total

    threshold, noise = None, None
    if args.maps or args.map_interactive:
        t_start = time.perf_counter()
        threshold, noise = fit_sigmoids(config.injection_voltage, efficiency)
        t_end = time.perf_counter()
        print(t_end - t_start)
        exit()

    ################################################################################

    if args.pixel:
        x, y = args.pixel
        efficiency_pixel = hits_signal[:,x,y] / config.injections_total
        fig, ax = plt.subplots()
        plot_single_scurve(ax, config.injection_voltage, efficiency_pixel)
        plt.show()

    ################################################################################

    if args.maps:
        assert threshold is not None and noise is not None
        map_types = args.maps
        if 'all' in map_types:
            map_types = map_choices[1:]
        # plot
        fig, axes = plt.subplots(1, len(map_types), squeeze=False, sharey=True, figsize=(6*len(map_types), 5))
        for ax, map_type in zip(axes[0], map_types):
            plot_single_map(ax, threshold, noise, pixel_edges, pixel_pos, cast(Any, map_type))
        plt.show()

    ################################################################################
    
    if args.map_interactive:
        assert threshold is not None and noise is not None
        fig, axes = plt.subplots(1, 3)
        # plot static maps
        plot_single_map(axes[0], threshold, noise, pixel_edges, pixel_pos, 'threshold')
        plot_single_map(axes[1], threshold, noise, pixel_edges, pixel_pos, 'noise')
        # plot dynamic data
        idx = (..., 0, 0)
        x_fit = np.linspace(np.min(config.injection_voltage), np.max(config.injection_voltage), 200)
        y_fit = fitfunc_sigmoid(x_fit, threshold[idx[1:]], noise[idx[1:]])
        line_data, = axes[2].plot(config.injection_voltage, efficiency[idx], 'x')
        line_fit, = axes[2].plot(x_fit, y_fit, 'r')
        fix_idx = False

        def mouse_event(event: MouseEvent):
            global idx, fix_idx
            if event.inaxes != axes[0] and event.inaxes != axes[1]:
                return
            idx_new = (..., *map(lambda p: int(p + 0.5), (event.xdata, event.ydata)))
            if fix_idx or idx == idx_new:
                return
            idx = idx_new
            line_data.set_ydata(efficiency[idx])
            y_fit = fitfunc_sigmoid(x_fit, threshold[idx[1:]], noise[idx[1:]])
            line_fit.set_ydata(y_fit)
        
        def press_event(event: MouseEvent):
            global fix_idx
            fix_idx = not fix_idx
        
        fig.canvas.mpl_connect('motion_notify_event', mouse_event)
        fig.canvas.mpl_connect('button_press_event', press_event)

        plt.ion()
        plt.show(block=True)
