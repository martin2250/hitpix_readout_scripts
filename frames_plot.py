#!/usr/bin/env python3
import argparse

import h5py
from matplotlib.backend_bases import MouseEvent
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

import frames.io
import util.gridscan

################################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    a_input_file = parser.add_argument(
        'input_file',
        help='h5 input file',
    )

    try:
        import argcomplete
        from argcomplete.completers import FilesCompleter
        setattr(a_input_file, 'completer', FilesCompleter('h5'))
        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args()

    ################################################################################
    # load data

    with h5py.File(args.input_file) as file:
        # get information about parameter scan
        if 'scan' in file:
            group_scan = file['scan']
            assert isinstance(group_scan, h5py.Group)
            scan_parameters, scan_shape = util.gridscan.load_scan(group_scan)
        else:
            scan_parameters, scan_shape = [], ()

        # load first dataset to get data shape
        # old data format used 'scurve' prefix
        prefix = 'frames'
        if not (prefix + '_0' * len(scan_shape)) in file:
            prefix = 'scurve'
        group_frames = file[prefix + '_0' * len(scan_shape)]
        assert isinstance(group_frames, h5py.Group)
        config, hit_frames_first, _ = frames.io.load_frames(group_frames)
        # create full data array
        hits_frames = np.zeros(scan_shape + hit_frames_first.shape)
        # store first scurve
        hits_frames[tuple(0 for _ in scan_shape) + (...,)] = hit_frames_first
        # store remaining scurves
        for idx in np.ndindex(*scan_shape):
            # do not load zeroth frame again
            if not any(idx):
                continue
            group_name = prefix + '_' + '_'.join(str(i) for i in idx)
            group_frame = file[group_name]
            assert isinstance(group_frame, h5py.Group)
            _, hits_frames_group, _ = frames.io.load_frames(group_frame)
            hits_frames[idx] = hits_frames_group

    ################################################################################
    # calculate pixel properties

    hits_frames = hits_frames.sum(axis=-3)

    fig = plt.gcf()
    gs = fig.add_gridspec(2, 2)

    # plot layout
    ax_map = fig.add_subplot(gs[1, 0])  # hit map for slider settings
    ax_hist = fig.add_subplot(gs[0, 0])  # histogram of hit map
    # total hits over selected slider setting
    ax_curve = fig.add_subplot(gs[0, 1])
    gs_sliders = gs[1, 1]                # space for sliders

    # interactive state
    idx_scan = tuple(0 for _ in scan_shape)
    id_slider = 0

    # add sliders
    sliders = []
    slider_bb = gs_sliders.get_position(fig)
    slider_height = (slider_bb.y1 - slider_bb.y0) / (len(scan_shape) + 1)
    def ax_slider(i: int):
        return fig.add_axes([
            slider_bb.x0,
            slider_bb.y0 + slider_height * i,
            slider_bb.x1 - slider_bb.x0,
            slider_height,
        ])
    for i_param, param in enumerate(scan_parameters):
        slider = Slider(
            ax=ax_slider(i_param),
            label=param.name,
            valmin=param.values[0],
            valmax=param.values[-1],
            valinit=param.values[0],
            valstep=param.values,
        )
        slider.parameter_id = i_param
        sliders.append(slider)
    slider_range = Slider(
        ax=ax_slider(len(scan_shape)),
        label='range',
        valmin=0.0,
        valmax=1.0,
        valinit=1.0,
    )

    # data ranges
    range_hits_full = max(np.min(hits_frames), 10), np.max(hits_frames)
    range_hits = range_hits_full

    # plot histograms
    ax_hist.set_xlabel('Hits')
    _, bins_hits, bars_hits = ax_hist.hist(
        hits_frames[idx_scan].flat, bins=30, range=range_hits)

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
    ax_curve.set_ylabel('Total Hits')
    ax_curve.set_yscale('log')

    def redraw_curve():
        if len(scan_shape) == 0:
            return
        sum_axes = list(range(len(hits_frames.shape)))
        del sum_axes[id_slider]
        data_x = scan_parameters[id_slider].values
        data_y = np.sum(hits_frames, axis=tuple(sum_axes))
        line_data.set_xdata(data_x)
        line_data.set_ydata(data_y)
        ax_curve.set_xlim(np.min(data_x), np.max(data_x))
        ax_curve.set_ylim(np.min(data_y), np.max(data_y))
        ax_curve.set_xlabel(scan_parameters[id_slider].name)
    redraw_curve()

    # add change handler
    def slider_on_changed(_):
        global idx_scan, sliders, scan_parameters
        idx_new = []
        for slider, param in zip(sliders, scan_parameters):
            idx_new.append(np.argmax(param.values == slider.val))
        idx_new = tuple(idx_new)
        if idx_new == idx_scan:
            return
        idx_scan = idx_new
        redraw_hists()
        redraw_maps()
        redraw_curve()

    for slider in sliders:
        slider.on_changed(slider_on_changed)

    def press_event(event: MouseEvent):
        global id_slider
        for i_slider, slider in enumerate(sliders):
            if event.inaxes == slider.ax:
                id_slider = i_slider

    fig.canvas.mpl_connect('button_press_event', press_event)

    ############################################################################
    
    def slider_range_onchanged(_):
        global range_hits, bins_hits, bars_hits
        log_min = np.log(range_hits_full[0])
        log_max = np.log(range_hits_full[1])
        log_new = log_min + slider_range.val * (log_max - log_min)
        range_hits = (range_hits_full[0], np.exp(log_new))
        # update map
        im_hits.set(clim=range_hits)
        # update histogram
        ax_hist.clear()
        _, bins_hits, bars_hits = ax_hist.hist(
            hits_frames[idx_scan].flat, bins=30, range=range_hits)
        ax_hist.set_xlabel('Hits')

    slider_range.on_changed(slider_range_onchanged)

    # plot
    plt.ion()
    plt.show(block=True)
