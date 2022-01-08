#!/usr/bin/env python3
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, cast

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent, KeyEvent, MouseButton
from matplotlib.widgets import Slider
from matplotlib.colors import LogNorm

import scurve.analysis
import laser_scurve.io
import util.gridscan

# TODO: add noise / dead pixel markers
# TODO: limit curve points at efficiency one

################################################################################
# multithreaded fitting


@dataclass
class FitJob:
    injection_voltage: np.ndarray
    efficiency: np.ndarray
    idx: tuple


################################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    a_input_file = parser.add_argument(
        'input_file',
        help='h5 input file',
    )

    parser.add_argument(
        '--no_fit',
        action='store_true',
        help='do not fit curves (no threshold/noise map)',
    )

    try:
        import argcomplete
        from argcomplete.completers import FilesCompleter
        setattr(a_input_file, 'completer', FilesCompleter('h5'))
        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args()

    ############################################################################
    # load data

    with h5py.File(args.input_file) as file:
        # get information about parameter scan
        group_scan = file['scan']
        assert isinstance(group_scan, h5py.Group)
        scan_parameters, scan_shape = util.gridscan.load_scan(group_scan)
        # load first dataset to get data shape
        prefix = 'laser_scurve'
        group_scurve = file[
            util.gridscan.group_name(prefix, (0 for _ in scan_shape))
        ]
        assert isinstance(group_scurve, h5py.Group)
        config, frames_first = laser_scurve.io.load_laser_scurves(group_scurve)
        # create full data array
        frames = np.zeros(scan_shape + frames_first.shape)
        # store remaining scurves
        print('reading file')
        for idx in np.ndindex(*scan_shape):
            group_scurve = file[util.gridscan.group_name(prefix, idx)]
            assert isinstance(group_scurve, h5py.Group)
            _, frames_group = laser_scurve.io.load_laser_scurves(group_scurve)
            frames[idx] = frames_group

    ############################################################################
    # prepare data for plotting

    # add pixel position to scan parameters
    scan_parameters = scan_parameters + [
        util.gridscan.ParameterScan('pixel_y', np.arange(frames.shape[-2])),
        util.gridscan.ParameterScan('pixel_x', np.arange(frames.shape[-1])),
    ]
    scan_shape = scan_shape + (frames.shape[-2], frames.shape[-1])

    # move threshold offset axis to end
    # before: [scan]...[scan][thresh][pixel_y][pixel_x]
    # after:  [scan]...[scan][pixel_y][pixel_x][thresh]
    frames = np.moveaxis(frames, -3, -1)
    efficiency = frames / config.injections_total

    print(f'{frames.shape=} {frames.size=}')

    # result arrays
    threshold, noise = np.zeros(frames.shape[:-1]), np.zeros(frames.shape[:-1])

    if not args.no_fit:
        print('fitting curves')
        # fit sigmoid expects higher efficiency for  higher x
        offset_voltage = 4.2
        mock_injection_voltage = offset_voltage - config.threshold_offsets
        # reverse to make array ascending
        mock_injection_voltage = mock_injection_voltage[::-1]

        # use process for each pixel column -> not too many processes
        def fit_job(index_x: int) -> tuple[np.ndarray, np.ndarray]:
            res_th = np.zeros(frames.shape[:-2])
            res_no = np.zeros(frames.shape[:-2])
            for idx in np.ndindex(*frames.shape[:-2]):
                efficiency_loc = efficiency[idx + (index_x,)]
                t, w = scurve.analysis.fit_sigmoid(
                    mock_injection_voltage,
                    efficiency_loc[::-1],  # also reverse here
                )
                res_th[idx] = offset_voltage - t
                res_no[idx] = w
            return res_th, res_no

        results = ProcessPoolExecutor().map(fit_job, range(frames.shape[-2]))

        # store in final results array
        for index_x, result in tqdm.tqdm(
            enumerate(results),
            total=frames.shape[-2],
            dynamic_ncols=True,
        ):
            threshold[(..., index_x)], noise[(..., index_x)] = result

    ############################################################################

    fig = plt.gcf()
    gs = fig.add_gridspec(2, 3)

    # plot layout
    ax_rawframe = fig.add_subplot(gs[0, 0])
    ax_thresh = fig.add_subplot(gs[0, 1])
    ax_curve = fig.add_subplot(gs[0, 2])
    ax_hits = fig.add_subplot(gs[1, 0])
    ax_noise = fig.add_subplot(gs[1, 1])
    gs_sliders = gs[1, 2]

    # interactive state
    # parameter indices to show in ax_thresh/ax_noise
    im_axis_idx_x = 0
    im_axis_idx_y = 1
    # which pixel (of ax_thresh/ax_noise) to show in ax_curve
    idx_im_pixel = (0, 0)
    # slider settings
    idx_sliders = tuple(0 for _ in scan_shape)

    # add sliders
    sliders = []
    slider_bb = gs_sliders.get_position(fig)
    slider_height = (slider_bb.y1 - slider_bb.y0) / (len(scan_shape) + 1)
    for i_slider, param in enumerate(scan_parameters):
        ax_slider = fig.add_axes([
            slider_bb.x0,
            slider_bb.y0 + slider_height * i_slider,
            slider_bb.x1 - slider_bb.x0,
            slider_height,
        ])
        label = param.name
        if param.values[0] > param.values[-1]:
            label += ' (rev)'
        sliders.append(Slider(
            ax=ax_slider,
            label=label,
            valmin=np.min(param.values),
            valmax=np.max(param.values),
            valinit=param.values[0],
            valstep=param.values,
        ))

    # data ranges
    threshold_clean = threshold[
        np.isfinite(threshold) &
        (threshold > -0.5) &
        (threshold < 2)
    ]
    noise_clean = noise[
        np.isfinite(noise) &
        (noise > 0) &
        (noise < 1000)
    ]

    if not args.no_fit:
        range_threshold = np.min(threshold_clean), np.max(threshold_clean)
        range_noise = np.min(noise_clean), np.max(noise_clean)
    else:
        range_threshold = range_noise = 0, 1

    # # plot raw frames
    # def rawframe_data_get() -> np.ndarray:
    #     axes_sum = list(range(frames.ndim))
    #     del axes_sum[-2]
    #     del axes_sum[-2]
    #     return np.log(1. + np.std(frames, axis=tuple(axes_sum)))

    im_raw_frame = ax_rawframe.imshow(
        np.ones(frames.shape[-3:-1]),
        norm=LogNorm(),
    )
    ax_rawframe.set_xlabel('pixel_x')
    ax_rawframe.set_ylabel('pixel_y')
    fig.colorbar(im_raw_frame, ax=ax_rawframe)
    rawframe_idx = 0

    rawframe_data = np.sum(frames, axis=-1)

    def rawframe_update():
        global rawframe_idx
        # which axes should be reduced?
        axes_sum = tuple(range(rawframe_data.ndim - 2))
        data_new = None
        # which visualization?
        match rawframe_idx:
            case 0:
                data_new = np.max(rawframe_data, axis=axes_sum) - np.min(rawframe_data, axis=axes_sum)
                ax_rawframe.set_title('Raw Sensor Hits (max-min)')
            case 1:
                data_new = np.sum(rawframe_data, axis=axes_sum)
                ax_rawframe.set_title('Raw Sensor Hits (sum)')
            case 2:
                data_new = np.std(rawframe_data, axis=axes_sum)
                ax_rawframe.set_title('Raw Sensor Hits (std)')
        # update plot
        assert data_new is not None
        data_new += 1.
        im_raw_frame.set_data(data_new)
        im_raw_frame.set_norm(LogNorm(np.min(data_new), np.max(data_new)))
        
        rawframe_idx += 1
        if rawframe_idx > 2:
            rawframe_idx = 0

    rawframe_update()

    # plot images
    def image_take(source: np.ndarray) -> np.ndarray:
        assert im_axis_idx_x != im_axis_idx_y
        index: list[Any] = list(idx_sliders)
        index[im_axis_idx_x] = slice(None)
        index[im_axis_idx_y] = slice(None)
        image = source[tuple(index)]
        if im_axis_idx_x < im_axis_idx_y:
            image = np.swapaxes(image, 0, 1)
        return image

    def image_extent() -> tuple[float, float, float, float]:
        sp_x = scan_parameters[im_axis_idx_x].values
        sp_y = scan_parameters[im_axis_idx_y].values
        step_x = sp_x[1] - sp_x[0]
        step_y = sp_y[1] - sp_y[0]
        left = sp_x[0] - step_x / 2
        right = sp_x[-1] + step_x / 2
        bottom = sp_y[-1] + step_y / 2
        top = sp_y[0] - step_y / 2
        return left, right, bottom, top

    im_thresh = ax_thresh.imshow(
        image_take(threshold),
        vmin=range_threshold[0],
        vmax=range_threshold[1],
        extent=image_extent(),
        aspect='auto',
    )
    im_noise = ax_noise.imshow(
        1e3*image_take(noise),
        vmin=range_noise[0],
        vmax=range_noise[1],
        extent=image_extent(),
        aspect='auto',
    )
    im_hits = ax_hits.imshow(
        np.sum(image_take(frames), axis=-1),
        extent=image_extent(),
        aspect='auto',
    )
    ax_thresh.set_title('Threshold (V)')
    ax_noise.set_title('Noise (mV)')

    fig.colorbar(im_thresh, ax=ax_thresh)
    fig.colorbar(im_noise, ax=ax_noise)
    fig.colorbar(im_hits, ax=ax_hits)

    def images_redraw():
        im_thresh.set_data(image_take(threshold))
        im_noise.set_data(1e3*image_take(noise))
        hits = np.sum(image_take(frames), axis=-1)
        im_hits.set_data(hits)
        im_hits.set_clim(np.min(hits), np.max(hits))
        extent = image_extent()
        im_thresh.set_extent(extent)
        im_noise.set_extent(extent)
        im_hits.set_extent(extent)
        for ax in (ax_thresh, ax_noise, ax_hits):
            ax.set_xlabel(scan_parameters[im_axis_idx_x].name)
            ax.set_ylabel(scan_parameters[im_axis_idx_y].name)
    images_redraw()

    # plot scurve
    curve_x_fit = np.linspace(
        np.min(config.threshold_offsets),
        np.max(config.threshold_offsets),
        200,
    )
    curve_line_data, = ax_curve.plot(
        config.threshold_offsets,
        config.threshold_offsets,
        'x',
    )
    curve_line_fit, = ax_curve.plot(curve_x_fit, curve_x_fit, 'r-')
    ax_curve.set_ylim(0, 1.05)
    ax_curve.set_xlabel('Threshold Offset (V)')
    ax_curve.set_ylabel('Efficiency')

    def curve_redraw():
        curve_line_data.set_ydata(efficiency[idx_sliders])
        # fit
        t, n = threshold[idx_sliders], noise[idx_sliders]
        ydata = scurve.analysis.fitfunc_sigmoid(curve_x_fit, t, -n)
        curve_line_fit.set_ydata(ydata)
    curve_redraw()

    # add change handler

    def slider_on_changed(*_):
        global idx_sliders, sliders, scan_values
        idx_new = []
        for slider, param in zip(sliders, scan_parameters):
            idx_new.append(np.argmax(param.values == slider.val))
        idx_new = tuple(idx_new)
        if idx_new == idx_sliders:
            return
        idx_sliders = idx_new
        images_redraw()
        curve_redraw()

    for slider in sliders:
        slider.on_changed(slider_on_changed)

    def button_press_event(event: MouseEvent):
        global idx_sliders
        if event.inaxes == ax_rawframe:
            if event.button is MouseButton.LEFT:
                pixel_new = tuple(map(
                    lambda p: int(p + 0.5),
                    (event.xdata, event.ydata),
                ))
                if pixel_new == idx_sliders[-2:]:
                    return
                idx_sliders = idx_sliders[:-2] + pixel_new
                sliders[-1].set_val(pixel_new[0])
                sliders[-2].set_val(pixel_new[1])
                images_redraw()
            else:
                rawframe_update()
            return
        if event.inaxes in (ax_thresh, ax_noise, ax_hits):
            idx_x_new = int(np.argmin(np.abs(
                event.xdata - scan_parameters[im_axis_idx_x].values
            )))
            idx_y_new = int(np.argmin(np.abs(
                event.ydata - scan_parameters[im_axis_idx_y].values
            )))
            idx_sliders_l = list(idx_sliders)
            idx_sliders_l[im_axis_idx_x] = idx_x_new
            idx_sliders_l[im_axis_idx_y] = idx_y_new
            sliders[im_axis_idx_x].set_val(
                scan_parameters[im_axis_idx_x].values[idx_x_new]
            )
            sliders[im_axis_idx_y].set_val(
                scan_parameters[im_axis_idx_y].values[idx_y_new]
            )
            # images_redraw() not needed, image stays the same
            curve_redraw()
            return

    def key_press_event(event: KeyEvent):
        global im_axis_idx_x, im_axis_idx_y
        if event.key in 'xy':
            # mouse over which slider?
            for i_slider, slider in enumerate(sliders):
                if event.inaxes == slider.ax:
                    break
            else:
                return
            # update axes
            if event.key == 'x':
                if i_slider == im_axis_idx_y:
                    im_axis_idx_y = im_axis_idx_x
                im_axis_idx_x = i_slider
            if event.key == 'y':
                if i_slider == im_axis_idx_x:
                    im_axis_idx_x = im_axis_idx_y
                im_axis_idx_y = i_slider
            print(
                f'set {event.key.upper()} axis to {scan_parameters[i_slider].name}')
            images_redraw()

    # fig.canvas.mpl_connect('motion_notify_event', mouse_event)
    fig.canvas.mpl_connect('button_press_event', button_press_event)
    fig.canvas.mpl_connect('key_press_event', key_press_event)

    # plot
    plt.ion()
    plt.show(block=True)
