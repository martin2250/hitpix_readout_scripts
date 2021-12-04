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

import plot_scurves
import scurves

# TODO: add noise / dead pixel markers
# TODO: limit curve points at efficiency one

################################################################################
# multithreaded fitting


@dataclass
class FitJob:
    injection_voltage: np.ndarray
    efficiency: np.ndarray


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
        
        
        # scan_names = cast(list[str], group_scan.attrs['scan_names'])
        # scan_values = cast(list[list[float]], group_scan.attrs['scan_values'])
        
        
        scan_names = group_scan.attrs['scan_names']
        assert isinstance(scan_names, np.ndarray)
        scan_values: list[np.ndarray] = []
        for name in scan_names:
            dset_values = group_scan[name]
            assert isinstance(dset_values, h5py.Dataset)
            print(dset_values)
            values_f = dset_values[:]
            assert isinstance(values_f, np.ndarray)
            scan_values.append(values_f)

        print(scan_names)
        print(scan_values)


        scan_shape = tuple(len(values) for values in scan_values)
        # get info about injection scan
        group_scurve = file['scurve' + '_0' * len(scan_shape)]
        assert isinstance(group_scurve, h5py.Group)
        config, hits_signal_first, _ = scurves.load_scurve(group_scurve)
        # create full data array
        hits_signal = np.zeros(scan_shape + hits_signal_first.shape)
        # store first scurve
        hits_signal[tuple(0 for _ in scan_shape) + (...,)] = hits_signal_first
        # store remaining scurves
        def scan_indices(): return np.ndindex(cast(SupportsIndex, scan_shape))
        for idx in scan_indices():
            # do not load zeroth scurve
            if not any(idx):
                continue
            group_name = 'scurve_' + '_'.join(str(i) for i in idx)
            group_scurve = file[group_name]
            assert isinstance(group_scurve, h5py.Group)
            _, hits_signal_group, _ = scurves.load_scurve(group_scurve)
            hits_signal[idx] = hits_signal_group

    ################################################################################

    sensor_size = hits_signal_first.shape[1:]
    # pixel edges for pcolormesh
    pixel_edges = pixel_edges = cast(tuple[np.ndarray, np.ndarray], tuple(
        np.arange(size + 1) - 0.5 for size in sensor_size))
    # pixel indices
    pixel_pos = np.meshgrid(*(np.array(np.arange(size))
                            for size in sensor_size))

    ################################################################################
    # calculate pixel properties

    # shape of result
    shape_res = list(hits_signal.shape)
    del shape_res[-3]  # injection voltage axis
    shape_res = tuple(shape_res)

    # result arrays
    threshold, noise = np.zeros(shape_res), np.zeros(shape_res)
    efficiency = hits_signal / config.injections_total

    # each set of parameters is a separate job
    fit_jobs = (FitJob(config.injection_voltage,
                efficiency[idx]) for idx in scan_indices())

    def fit_job(job: FitJob):
        return plot_scurves.fit_sigmoids(job.injection_voltage, job.efficiency)
    results = ProcessPoolExecutor().map(fit_job, fit_jobs)

    # store in final results array
    for idx, result in tqdm.tqdm(zip(scan_indices(), results)):
        threshold[idx], noise[idx] = result

    # convert noise to mV
    noise *= 1e3

    ################################################################################

    fig = plt.gcf()
    gs = fig.add_gridspec(2, 3)

    # plot layout
    ax_hthresh = fig.add_subplot(gs[0, 0])
    ax_hnoise = fig.add_subplot(gs[1, 0])
    ax_thresh = fig.add_subplot(gs[0, 1])
    ax_noise = fig.add_subplot(gs[1, 1])
    ax_curve = fig.add_subplot(gs[0, 2])
    gs_sliders = gs[1, 2]

    # interactive state
    idx_pixel = (0, 0)
    idx_scan = tuple(0 for _ in scan_shape)
    fix_idx = False

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
    threshold_clean = threshold[np.isfinite(threshold)]
    noise_clean = noise[np.isfinite(noise)]

    range_threshold = np.min(threshold_clean), np.max(threshold_clean)
    range_noise = np.min(noise_clean), np.max(noise_clean)

    # plot histograms
    _, bins_threshold, bars_threshold = ax_hthresh.hist(
        threshold[idx_scan].flat, bins=30, range=range_threshold)
    _, bins_noise, bars_noise = ax_hnoise.hist(
        noise[idx_scan].flat, bins=30, range=range_noise)
    
    ax_hthresh.set_xlabel('Threshold (V)')
    ax_hnoise.set_xlabel('Noise (mV)')

    def redraw_hists():
        data_threshold, _ = np.histogram(
            threshold[idx_scan].flat, bins=bins_threshold)
        data_noise, _ = np.histogram(noise[idx_scan].flat, bins=bins_noise)
        for value, bar in zip(data_threshold, bars_threshold):
            bar.set_height(value)
        for value, bar in zip(data_noise, bars_noise):
            bar.set_height(value)
        ax_hthresh.set_ylim(0, np.max(data_threshold))
        ax_hnoise.set_ylim(0, np.max(data_noise))

    # plot maps
    im_thresh = ax_thresh.imshow(
        threshold[idx_scan].T,
        vmin=range_threshold[0],
        vmax=range_threshold[1],
    )
    im_noise = ax_noise.imshow(
        noise[idx_scan].T,
        vmin=range_noise[0],
        vmax=range_noise[1],
    )
    ax_thresh.set_title('Threshold (V)')
    ax_noise.set_title('Noise (mV)')

    fig.colorbar(im_thresh, ax=ax_thresh)
    fig.colorbar(im_noise, ax=ax_noise)

    def redraw_maps():
        im_thresh.set_data(threshold[idx_scan].T)
        im_noise.set_data(noise[idx_scan].T)

    # plot scurve
    x_fit = np.linspace(np.min(config.injection_voltage),
                        np.max(config.injection_voltage), 200)
    line_data, = ax_curve.plot(
        config.injection_voltage, config.injection_voltage, 'x')
    line_fit, = ax_curve.plot(x_fit, x_fit, 'r')
    ax_curve.set_ylim(0, 1)
    ax_curve.set_xlabel('Injection Voltage (V)')
    ax_curve.set_ylabel('Efficiency')

    def redraw_curve():
        y_fit = plot_scurves.fitfunc_sigmoid(
            x_fit,
            threshold[idx_scan + idx_pixel],
            1e-3*noise[idx_scan + idx_pixel]
        )
        line_data.set_ydata(efficiency[idx_scan+(...,)+idx_pixel])
        line_fit.set_ydata(y_fit)
    redraw_curve()

    # add change handler

    def slider_on_changed(*_):
        global idx_scan, sliders, scan_values
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

    # pixel selection
    lines_selected = [ax.plot([0], [0], 'rx')[0]
                      for ax in (ax_thresh, ax_noise)]

    def mouse_event(event: MouseEvent):
        global idx_pixel
        if fix_idx or (event.inaxes != ax_thresh and event.inaxes != ax_noise):
            return
        idx_pixel_new = tuple(
            map(lambda p: int(p + 0.5), (event.xdata, event.ydata)))
        if idx_pixel == idx_pixel_new:
            return
        idx_pixel = idx_pixel_new
        for line in lines_selected:
            line.set_xdata(idx_pixel[:1])
            line.set_ydata(idx_pixel[1:])
        redraw_curve()

    def press_event(event: MouseEvent):
        global fix_idx
        if event.inaxes != ax_thresh and event.inaxes != ax_noise:
            return
        fix_idx = not fix_idx

    fig.canvas.mpl_connect('motion_notify_event', mouse_event)
    fig.canvas.mpl_connect('button_press_event', press_event)

    # plot
    plt.ion()
    plt.show(block=True)
