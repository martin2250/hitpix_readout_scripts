#!/usr/bin/env python3
import argparse
from typing import Any
import tqdm
import h5py
from matplotlib.backend_bases import MouseEvent
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import ampout_snr.analysis
from ampout_snr.io import load_ampout_snr
import util.gridscan
from concurrent.futures import ProcessPoolExecutor

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
    pex = ProcessPoolExecutor()

    with h5py.File(args.input_file) as file:
        # get information about parameter scan
        if 'scan' in file:
            group_scan = file['scan']
            assert isinstance(group_scan, h5py.Group)
            scan_parameters, scan_shape = util.gridscan.load_scan(group_scan)
        else:
            scan_parameters, scan_shape = [], ()

        _, data_0, time_offset, time_delta = load_ampout_snr(
            file,
            'ampout_snr' + '_0' * len(scan_shape),
        )
        num_wfms = data_0.shape[0]
        len_wfm = data_0.shape[1]
        del data_0

        # use 80% of the trace leading up to the trigger event as baseline
        cnt_baseline = int(0.8 * time_offset / time_delta)

        def read_idx(idx: tuple[int, ...]) -> tuple[tuple[int, ...], np.ndarray]:
            dset_name = 'ampout_snr' + ''.join(f'_{i}' for i in idx)
            dset_snr = file[dset_name]
            assert isinstance(dset_snr, h5py.Dataset)
            data = dset_snr[()]
            assert isinstance(data, np.ndarray)
            if data.ndim == 3:
                data = data.reshape(data.shape[1:])
            results = np.zeros(data.shape[0])
            # iterate over all waveforms and find baseline + peak
            for i, wfm in enumerate(data):
                assert isinstance(wfm, np.ndarray)
                # baseline
                baseline = np.mean(wfm[:cnt_baseline])
                peak = ampout_snr.analysis.fit_peak(wfm)
                results[i] = peak - baseline
            return idx, results
        
        wfms_shape = scan_shape + (num_wfms,)
        y_peak = np.zeros(wfms_shape)
        
        for idx, data in tqdm.tqdm(
            pex.map(read_idx, np.ndindex(*scan_shape)),
            total=np.prod(scan_shape),
        ):
            y_peak[idx] = data


    for x in y_peak:
        plt.hist(x, 30)
        signal = np.mean(x)
        noise = np.std(x)
        print(f'{signal=}, {noise=} {signal/noise}')
    plt.show()

    # ################################################################################
    # # calculate waveform properties
    # wfms_shape = y_data.shape[:-1]

    # y_peak = np.zeros(wfms_shape)
    # y_baseline = np.zeros(wfms_shape)

    # for val, idx in tqdm.tqdm(zip(
    #     pex.map(ampout_snr.analysis.fit_curve, y_data.reshape(-1, y_data.shape[-1])),
    #     np.ndindex(*wfms_shape),
    #     )):
    #     y_peak[idx] = val

    # for idx in np.ndindex(*wfms_shape):
    #     wfm = y_data[idx]
    #     assert isinstance(wfm, np.ndarray)
    #     i_max = np.argmax(wfm)
    #     y_max = wfm[i_max]
    #     y_cut = y_max * 0.9
    #     i_right = np.argmax((wfm < y_cut) & (x_index > i_max))
    #     i_left = np.argmax((wfm[:i_max] < y_cut)*x_index[:i_max])

    #     y_fit = wfm[i_left:i_right]

    #     x_fit = np.arange(y_fit.size)
    #     def parabola(x, a, x0, c):
    #         return a - c * np.square(x - x0)

    #     popt, _ = scipy.optimize.curve_fit(
    #         f=parabola,
    #         xdata=x_fit,
    #         ydata=y_fit,
    #         p0=(y_max, i_max, 0.01),
    #     )

    #     y_peak[idx] = popt[0]

    # print(y_peak.shape)

    # plt.hist(y_peak[0], 30)

    # # for i in range(24):
    # #     plt.plot(x_time, y_data[i, 0, :])
    # plt.show()

    # hits_frames = hits_frames.sum(axis=-3)

    # fig = plt.gcf()
    # gs = fig.add_gridspec(2, 2)

    # # plot layout
    # ax_map = fig.add_subplot(gs[1, 0])  # hit map for slider settings
    # ax_hist = fig.add_subplot(gs[0, 0])  # histogram of hit map
    # # total hits over selected slider setting
    # ax_curve = fig.add_subplot(gs[0, 1])
    # gs_sliders = gs[1, 1]                # space for sliders

    # # interactive state
    # idx_scan = tuple(0 for _ in scan_shape)
    # id_slider = 0

    # # add sliders
    # sliders = []
    # slider_bb = gs_sliders.get_position(fig)
    # slider_height = (slider_bb.y1 - slider_bb.y0) / (len(scan_shape) + 1)
    # def ax_slider(i: int):
    #     return fig.add_axes([
    #         slider_bb.x0,
    #         slider_bb.y0 + slider_height * i,
    #         slider_bb.x1 - slider_bb.x0,
    #         slider_height,
    #     ])
    # for i_param, param in enumerate(scan_parameters):
    #     slider = Slider(
    #         ax=ax_slider(i_param),
    #         label=param.name if (param.values[0] < param.values[-1]) else f'{param.name} (rev)',
    #         valmin=np.min(param.values),
    #         valmax=np.max(param.values),
    #         valinit=param.values[0],
    #         valstep=param.values,
    #     )
    #     slider.parameter_id = i_param
    #     sliders.append(slider)
    # slider_range = Slider(
    #     ax=ax_slider(len(scan_shape)),
    #     label='range',
    #     valmin=0.0,
    #     valmax=1.0,
    #     valinit=1.0,
    # )
    # sliders[id_slider].label.set_backgroundcolor('lightgreen')

    # # data ranges
    # range_hits_full = max(np.min(hits_frames), 10), np.max(hits_frames)
    # range_hits = range_hits_full

    # # plot histograms
    # ax_hist.set_xlabel('Hits')
    # _, bins_hits, bars_hits = ax_hist.hist(
    #     hits_frames[idx_scan].flat, bins=30, range=range_hits)

    # def redraw_hists():
    #     data_hits, _ = np.histogram(
    #         hits_frames[idx_scan].flat, bins=bins_hits)
    #     for value, bar in zip(data_hits, bars_hits):
    #         bar.set_height(value)
    #     ax_hist.set_ylim(0, np.max(data_hits))

    # # plot maps
    # im_hits = ax_map.imshow(
    #     hits_frames[idx_scan].T,
    #     vmin=range_hits[0],
    #     vmax=range_hits[1],
    # )
    # ax_map.set_title('Hits')
    # fig.colorbar(im_hits, ax=ax_map)

    # def redraw_maps():
    #     im_hits.set_data(hits_frames[idx_scan].T)

    # # plot scurve
    # line_data, = ax_curve.plot([], [])
    # ax_curve.set_ylabel('Total Hits')
    # ax_curve.set_yscale('log')

    # def redraw_curve():
    #     if len(scan_shape) == 0:
    #         return
    #     idx_curve: list[Any] = list(idx_scan)
    #     idx_curve[id_slider] = Ellipsis
    #     data_y = np.sum(hits_frames[tuple(idx_curve)], axis=(-1, -2))
    #     line_data.set_ydata(data_y)
    #     data_x = scan_parameters[id_slider].values
    #     line_data.set_xdata(data_x)
    #     ax_curve.set_xlim(np.min(data_x), np.max(data_x))
    #     ax_curve.set_ylim(np.min(data_y), np.max(data_y))
    #     ax_curve.set_xlabel(scan_parameters[id_slider].name)
    # redraw_curve()

    # # add change handler
    # def slider_on_changed(_):
    #     global idx_scan, sliders, scan_parameters
    #     idx_new = []
    #     for slider, param in zip(sliders, scan_parameters):
    #         idx_new.append(np.argmax(param.values == slider.val))
    #     idx_new = tuple(idx_new)
    #     if idx_new == idx_scan:
    #         return
    #     idx_scan = idx_new
    #     redraw_hists()
    #     redraw_maps()
    #     redraw_curve()

    # for slider in sliders:
    #     slider.on_changed(slider_on_changed)

    # def press_event(event: MouseEvent):
    #     global id_slider
    #     # do not accept left click
    #     if event.button == 1:
    #         return
    #     for i_slider, slider in enumerate(sliders):
    #         if event.inaxes == slider.ax:
    #             sliders[id_slider].label.set_backgroundcolor('white')
    #             id_slider = i_slider
    #             sliders[id_slider].label.set_backgroundcolor('lightgreen')
    #             redraw_curve()
    #             return

    # fig.canvas.mpl_connect('button_press_event', press_event)

    # ############################################################################

    # def slider_range_onchanged(_):
    #     global range_hits, bins_hits, bars_hits
    #     log_min = np.log(range_hits_full[0])
    #     log_max = np.log(range_hits_full[1])
    #     log_new = log_min + slider_range.val * (log_max - log_min)
    #     range_hits = (range_hits_full[0], np.exp(log_new))
    #     # update map
    #     im_hits.set(clim=range_hits)
    #     # update histogram
    #     ax_hist.clear()
    #     _, bins_hits, bars_hits = ax_hist.hist(
    #         hits_frames[idx_scan].flat, bins=30, range=range_hits)
    #     ax_hist.set_xlabel('Hits')

    # slider_range.on_changed(slider_range_onchanged)

    # # plot
    # plt.ion()
    # plt.show(block=True)
