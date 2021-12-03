#!/usr/bin/python
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import SupportsIndex, cast

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.widgets import Slider

import frames

################################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'input_file',
        help='h5 input file',
    )

    args = parser.parse_args()

    ################################################################################
    # load data

    with h5py.File(args.input_file) as file:
        # get information about parameter scan
        group_scan = file['scan']
        assert isinstance(group_scan, h5py.Group)
        scan_names = cast(list[str], group_scan.attrs['scan_names'])
        scan_values = cast(list[list[float]], group_scan.attrs['scan_values'])
        scan_shape = tuple(len(values) for values in scan_values)
        # get info about injection scan
        group_frame_first = file['frames' + '_0' * len(scan_shape)]
        assert isinstance(group_frame_first, h5py.Group)
        config, hit_frames_first = frames.load_frames(group_frame_first)
        # create full data array
        hits_frames = np.zeros(scan_shape + hit_frames_first.shape)
        # store first scurve
        hits_frames[tuple(0 for _ in scan_shape) + (...,)] = hit_frames_first
        # store remaining scurves
        def scan_indices(): return np.ndindex(cast(SupportsIndex, scan_shape))
        for idx in scan_indices():
            # do not load zeroth scurve
            if not any(idx):
                continue
            group_name = 'frames_' + '_'.join(str(i) for i in idx)
            group_frame = file[group_name]
            assert isinstance(group_frame, h5py.Group)
            _, hits_frames_group = frames.load_frames(group_frame)
            hits_frames[idx] = hits_frames_group

    ################################################################################

    sensor_size = hit_frames_first.shape[1:]
    # pixel edges for pcolormesh
    pixel_edges = pixel_edges = cast(tuple[np.ndarray, np.ndarray], tuple(
        np.arange(size + 1) - 0.5 for size in sensor_size))
    # pixel indices
    pixel_pos = np.meshgrid(*(np.array(np.arange(size))
                            for size in sensor_size))

    ################################################################################
    # calculate pixel properties

    hits_frames = hits_frames.sum(axis=-3)

    fig = plt.gcf()
    gs = fig.add_gridspec(2, 2)

    # plot layout
    ax_map   = fig.add_subplot(gs[1, 0]) # hit map for slider settings
    ax_hist  = fig.add_subplot(gs[0, 0]) # histogram of hit map
    ax_curve = fig.add_subplot(gs[0, 1]) # total hits over selected slider setting
    gs_sliders = gs[1, 1]                # space for sliders

    # interactive state
    idx_scan = tuple(0 for _ in scan_shape)
    id_slider = 0

    # add sliders
    sliders = []
    slider_bb = gs_sliders.get_position(fig)
    slider_height = (slider_bb.y1 - slider_bb.y0) / len(scan_names)
    for i_slider, name, values in zip(range(1000), scan_names, scan_values):
        ax_slider = fig.add_axes([
            slider_bb.x0,
            slider_bb.y0 + slider_height * i_slider,
            slider_bb.x1 - slider_bb.x0,
            slider_height,
        ])
        sliders.append(Slider(
            ax=ax_slider,
            label=name,
            valmin=values[0],
            valmax=values[-1],
            valinit=values[0],
            valstep=values,
        ))

    # data ranges
    range_hits = np.min(hits_frames), np.percentile(hits_frames, 95)
    # range_hits = np.min(hits_frames), np.max(hits_frames)

    # plot histograms
    _, bins_hits, bars_hits = ax_hist.hist(
        hits_frames[idx_scan].flat, bins=30, range=range_hits)

    ax_hist.set_xlabel('Hits')

    def redraw_hists():
        data_hits, _ = np.histogram(
            hits_frames[idx_scan].flat, bins=bins_hits)
        for value, bar in zip(data_hits, bars_hits):
            bar.set_height(value)
        ax_hist.set_ylim(0, np.max(data_hits))

    # plot maps
    im_hits = ax_map.imshow(
        hits_frames[idx_scan].T,
        vmin=range_hits[0],
        vmax=range_hits[1],
    )
    ax_map.set_title('Hits')
    fig.colorbar(im_hits, ax=ax_map)

    def redraw_maps():
        im_hits.set_data(hits_frames[idx_scan].T)

    # plot scurve
    line_data, = ax_curve.plot([], [])
    ax_curve.set_xlabel(scan_names[id_slider])
    ax_curve.set_ylabel('Total Hits')

    def redraw_curve():
        sum_axes = list(range(len(hits_frames.shape)))
        del sum_axes[id_slider]
        data = np.sum(hits_frames, axis=tuple(sum_axes))
        line_data.set_xdata(scan_values[id_slider])
        line_data.set_ydata(data)
        ax_curve.set_xlim(np.min(scan_values[id_slider]), np.max(scan_values[id_slider]))
        ax_curve.set_ylim(np.min(data), np.max(data))
        ax_curve.set_xlabel(scan_names[id_slider])
    redraw_curve()

    # add change handler
    def slider_on_changed(changed_slider):
        global idx_scan, sliders, scan_values, id_slider
        for i_slider, slider in enumerate(sliders):
            if slider == changed_slider:
                id_slider = i_slider
        idx_new = []
        for slider, values in zip(sliders, scan_values):
            idx_new.append(np.argmax(values == slider.val))
        idx_new = tuple(idx_new)
        if idx_new == idx_scan:
            return
        idx_scan = idx_new
        redraw_hists()
        redraw_maps()
        redraw_curve()

    for slider in sliders:
        slider.on_changed(slider_on_changed)

    # plot
    plt.ion()
    plt.show(block=True)
