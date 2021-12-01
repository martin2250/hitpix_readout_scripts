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
import scurves
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent

import numpy as np

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

    args = parser.parse_args()

    ################################################################################
    # load data

    with h5py.File(args.input_file) as file:
        group_scan = file['scan']
        assert isinstance(group_scan, h5py.Group)
        scan_names = cast(list[str], group_scan['scan_names'])
        scan_values = np.ndarray(cast(Any, group_scan['scan_values']))
        scan_shape = []
        # load group
        group = file[h5group]
        assert isinstance(group, h5py.Group)
        config, hits_signal, hits_noise = scurves.load_scurve(group)
    
    sensor_size = hits_noise.shape[1:]
    # pixel edges for pcolormesh
    pixel_edges = tuple(np.arange(size + 1) - 0.5 for size in sensor_size)
    # pixel indices
    pixel_x, pixel_y = np.meshgrid(*(np.array(np.arange(size)) for size in sensor_size))

    # calculate pixel properties
    efficiency = hits_signal / config.injections_total
    threshold, noise = fit_sigmoids(config.injection_voltage, efficiency)

    ################################################################################

    if args.single_scurve:
        x, y = args.single_scurve
        efficiency_pixel = hits_signal[:,x,y] / config.injections_total
        fig, ax = plt.subplots()
        plot_single_scurve(ax, config.injection_voltage, efficiency_pixel)
        plt.show()

    ################################################################################

    if args.maps:
        map_types = args.maps
        if 'all' in map_types:
            map_types = map_choices[1:]
        # plot
        fig, axes = plt.subplots(1, len(map_types), squeeze=False, sharey=True, figsize=(6*len(map_types), 5))
        for ax, map_type in zip(axes[0], map_types):
            plot_single_map(ax, threshold, noise, cast(Any, map_type))
        plt.show()

    ################################################################################
    
    if args.map_interactive:
        fig, axes = plt.subplots(1, 3)
        # plot static maps
        plot_single_map(axes[0], threshold, noise, 'threshold')
        plot_single_map(axes[1], threshold, noise, 'noise')
        # plot dynamic data
        idx = (..., 0, 0)
        x_fit = np.linspace(np.min(config.injection_voltage), np.max(config.injection_voltage), 200)
        y_fit = fitfunc_sigmoid(x_fit, threshold[idx[1:]], noise[idx[1:]])
        line_data, = axes[2].plot(config.injection_voltage, efficiency[idx], 'x')
        line_fit, = axes[2].plot(x_fit, y_fit, 'r')

        def mouse_event(event: MouseEvent):
            global idx
            if event.inaxes != axes[0] and event.inaxes != axes[1]:
                return
            idx_new = (..., *map(lambda p: int(p + 0.5), (event.xdata, event.ydata)))
            if idx == idx_new:
                return
            idx = idx_new
            line_data.set_ydata(efficiency[idx])
            y_fit = fitfunc_sigmoid(x_fit, threshold[idx[1:]], noise[idx[1:]])
            line_fit.set_ydata(y_fit)
        
        fig.canvas.mpl_connect('motion_notify_event', mouse_event)

        plt.ion()
        plt.show(block=True)
