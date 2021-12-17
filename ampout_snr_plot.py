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

        num_wfm_examples = 10

        # use 80% of the trace leading up to the trigger event as baseline
        cnt_baseline = int(-0.8 * time_offset / time_delta)

        def read_idx(idx: tuple[int, ...]) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
            dset_name = 'ampout_snr' + ''.join(f'_{i}' for i in idx)
            dset_snr = file[dset_name]
            assert isinstance(dset_snr, h5py.Dataset)
            if dset_snr.shape != data_0.shape:
                return idx, np.full(data_0.shape[0], np.nan), np.zeros((num_wfm_examples, len_wfm))
            data = dset_snr[()]
            assert isinstance(data, np.ndarray)
            if data.ndim == 3:
                data = data.reshape(data.shape[1:])
            peaks = np.zeros(data.shape[0])
            # iterate over all waveforms and find baseline + peak
            for i, wfm in enumerate(data):
                assert isinstance(wfm, np.ndarray)
                try:
                    peaks[i] = ampout_snr.analysis.fit_peak(wfm)
                except:
                    pass
            # calculate baselines
            baselines = np.mean(data[:,:cnt_baseline], axis=1)
            # get results
            results = peaks - baselines
            examples = data[:num_wfm_examples] - baselines[:num_wfm_examples].reshape(-1, 1)
            return idx, results, examples
        
        y_peak = np.zeros(scan_shape + (num_wfms,))
        y_example = np.zeros(scan_shape + (num_wfm_examples, len_wfm,)) # store 5 example waveforms
        
        for idx, data, example in tqdm.tqdm(
            pex.map(read_idx, np.ndindex(*scan_shape)),
            total=np.prod(scan_shape),
            dynamic_ncols=True,
        ):
            y_peak[idx] = data
            y_example[idx] = example


    ################################################################################

    fig = plt.gcf()
    gs = fig.add_gridspec(2, 2)

    # plot layout
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_snr = fig.add_subplot(gs[0, 1])
    ax_example = fig.add_subplot(gs[1, 0])
    gs_sliders = gs[1, 1]                # space for sliders

    ################################################################################

    id_slider = 0
    idx_scan = tuple(0 for _ in scan_shape)

    ################################################################################
    # histograms of peak heights

    y_peak = np.ma.array(y_peak, mask=~np.isfinite(y_peak))

    y_peak_range = np.min(y_peak), np.max(y_peak)
    y_peak_numbins = int(3 * (y_peak_range[1] - y_peak_range[0]) / np.mean(np.std(y_peak, axis=-1)))
    y_peak_bins = np.linspace(*y_peak_range, y_peak_numbins+1)
    y_peak_centers = np.convolve(y_peak_bins, [0.5, 0.5], mode='valid')
    y_peak_widths = y_peak_bins[1] - y_peak_bins[0]
    y_peak_bars = []

    # calculate histograms over all wfms in each group
    y_peak_histograms = np.zeros(scan_shape + (y_peak_numbins,))
    for idx in np.ndindex(*scan_shape):
        y_peak_histograms[idx], _ = np.histogram(y_peak[idx], y_peak_bins)
    
    def plot_histograms(slider_id_changed: bool):
        # select data for plotting
        idx: Any = list(idx_scan)
        idx[id_slider] = slice(None)
        data_plot = y_peak_histograms[tuple(idx)]
        # make data 2d even if there is only one histogram
        hists_plot = np.reshape(data_plot, (-1, y_peak_numbins))
        # plot
        if slider_id_changed:
            # reset everything
            ax_hist.clear()
            ax_hist.set_title('Peak Height Distribution')
            ax_hist.set_xlabel('Peak Height (mV)')
            ax_hist.set_ylabel('# of Pulses')
            y_peak_bars.clear()
            # create histograms
            for hist in hists_plot:
                bars = ax_hist.bar(y_peak_centers, hist, width=y_peak_widths)
                y_peak_bars.append(bars)
        else:
            for hist, bars in zip(hists_plot, y_peak_bars):
                for value, bar in zip(hist, bars):
                    bar.set_height(value)
    
    plot_histograms(True)

    ################################################################################
    # SNR (histogram width / histogram pos)

    # calculate mean and std over all wfms in each group
    y_peak_mean = np.mean(y_peak, axis=-1)
    y_peak_std = np.std(y_peak, axis=-1)
    y_peak_snr = y_peak_mean / y_peak_std

    snr_range = np.min(y_peak_snr), np.max(y_peak_snr)
    
    def plot_snrs():
        idx: Any = list(idx_scan)
        idx[id_slider] = slice(None)
        data_plot = y_peak_snr[tuple(idx)]
        # plot
        ax_snr.clear()
        ax_snr.set_xlabel('SNR (peak height mean/std)')
        ax_snr.set_ylabel('# of Occurences')
        ax_snr.set_title('Signal to Noise Ratios')
        ax_snr.hist(data_plot, 15, range=snr_range)

    plot_snrs()

    ################################################################################
    # example waveforms

    example_time = 1e6*(np.arange(len_wfm) * time_delta + time_offset)
    example_lines = []
    for _ in range(num_wfm_examples):
        line_ex, = ax_example.plot(example_time, example_time)
        example_lines.append(line_ex)
    
    ax_example.set_ylim(np.min(y_example), np.max(y_example))
    ax_example.set_title('Example Waveforms')
    ax_example.set_xlabel('Time (µs)')
    ax_example.set_ylabel('Voltage (V)')

    def plot_examples():
        for line, data in zip(example_lines, y_example[idx_scan]):
            line.set_ydata(data)

    plot_examples()

    ################################################################################
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
            label=param.name if (param.values[0] < param.values[-1]) else f'{param.name} (rev)',
            valmin=np.min(param.values),
            valmax=np.max(param.values),
            valinit=param.values[0],
            valstep=param.values,
        )
        slider.parameter_id = i_param
        sliders.append(slider)
    sliders[id_slider].label.set_backgroundcolor('lightgreen')

    def slider_on_changed(_):
        global idx_scan, sliders, scan_parameters
        idx_new = []
        for slider, param in zip(sliders, scan_parameters):
            idx_new.append(np.argmax(param.values == slider.val))
        idx_new = tuple(idx_new)
        if idx_new == idx_scan:
            return
        idx_scan = idx_new
        plot_histograms(False)
        plot_snrs()
        plot_examples()

    for slider in sliders:
        slider.on_changed(slider_on_changed)

    def press_event(event: MouseEvent):
        global id_slider
        # do not accept left click
        if event.button == 1:
            return
        for i_slider, slider in enumerate(sliders):
            if event.inaxes == slider.ax:
                sliders[id_slider].label.set_backgroundcolor('white')
                id_slider = i_slider
                sliders[id_slider].label.set_backgroundcolor('lightgreen')
                plot_histograms(True)
                plot_snrs()
                plot_examples()
                return

    fig.canvas.mpl_connect('button_press_event', press_event)

    plt.ion()
    plt.show(block=True)
