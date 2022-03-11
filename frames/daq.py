import math
import queue
import threading
import time
from typing import Optional

import h5py
import numpy as np
from rich import print
from rich.progress import Progress, TaskID
from hitpix import HitPixSetup
from hitpix.readout import HitPixReadout
from readout.fast_readout import FastReadout
from readout.instructions import Finish
from readout.sm_prog import decode_column_packets, prog_dac_config
from util.time_sync import TimeSync

from .io import FrameConfig
from .sm_prog import prog_read_adders, prog_read_frames


def _decode_responses(
    response_queue: queue.Queue[bytes],
    frames: list[np.ndarray],
    adders: list[np.ndarray],
    timestamps: list[np.ndarray],
    timeout: float,
    setup: HitPixSetup,
    reset_counters: bool,
    read_adders: bool,
    num_rows: int,
    evt_stop: threading.Event,
    num_runs: Optional[int] = None,
    callback=None,
    progress: Optional[tuple[Progress, TaskID]] = None,
    # frames, adders, times
    h5data: Optional[tuple[Optional[h5py.Dataset],
                           Optional[h5py.Dataset], h5py.Dataset]] = None,
):
    ctr_max = 1 << setup.chip.bits_counter
    hits_last = None
    # total number of frames received so far
    frames_total = 0

    mask_adders = (1 << setup.chip.bits_adder) - 1
    mask_counters = (1 << setup.chip.bits_counter) - 1

    if num_runs is None:
        # infinite iterator
        run_iter = iter(int, 1)
    else:
        run_iter = range(num_runs)

    # data processing loop
    for _ in run_iter:
        try:
            data = response_queue.get(True, timeout)
        except Exception as e:
            if evt_stop.is_set():
                break
            else:
                raise e
        if (qs := response_queue.qsize()) > 10:
            print(f'[red] queue overflowing: {qs}')
        # decode hits
        block_timestamps, block_frames = decode_column_packets(
            data,
            setup.pixel_columns,
            setup.chip.bits_adder,
            None,  # setup.chip.bits_counter,
        )
        block_frames = block_frames.reshape(
            -1,
            read_adders + num_rows,
            setup.pixel_columns,
        )
        block_numframes = block_frames.shape[0]

        # extract relevant timestamps
        block_timestamps = block_timestamps[::read_adders + num_rows]

        # extract adder values from first row
        block_adders = np.empty(0, dtype=np.uint32)
        if read_adders:
            block_adders = block_frames[:, 0, :]
            if num_rows == 0:
                block_frames = np.empty(0, dtype=np.uint32)
            else:
                block_frames = block_frames[:, 1:, :]
        # apply bit masks
        block_adders = np.bitwise_and(block_adders, mask_adders)
        block_frames = np.bitwise_and(block_frames, mask_counters)

        # counters not reset? apply diff between frames
        # TODO: probably remove this
        if not reset_counters:
            if hits_last is None:
                # duplicate first frame -> zero frame at start
                # this makes sure the total number of frames is
                # the same as with reset_counters=True
                hits_last = block_frames[:1]
            hits_last_next = block_frames[-1:]
            block_frames = np.diff(block_frames, axis=0, prepend=hits_last)
            hits_last = hits_last_next

        # counter counts down
        block_frames = (ctr_max - block_frames) % ctr_max

        if callback:
            callback(block_frames)

        if h5data is not None:
            dset_frames, dset_adders, dset_times = h5data
            size_new = frames_total+block_numframes
            slice_new = slice(frames_total, None)
            # store timestamps
            dset_times.resize(size_new, 0)
            dset_times[slice_new] = block_timestamps
            # store frames
            if dset_frames is not None:
                dset_frames.resize(size_new, 0)
                dset_frames[slice_new] = block_frames
            # store adders
            if dset_adders is not None:
                dset_adders.resize(size_new, 0)
                dset_adders[slice_new] = block_adders
        else:
            timestamps.append(block_timestamps)
            if num_rows > 0:
                frames.append(block_frames)
            if read_adders:
                adders.append(block_adders)


        frames_total += block_numframes
        if progress is not None:
            prog, task = progress
            prog.update(task, advance=block_numframes)


def read_frames(
    ro: HitPixReadout,
    fastreadout: FastReadout,
    config: FrameConfig,
    progress: Optional[tuple[Progress, TaskID]] = None,
    callback=None,
    evt_stop: Optional[threading.Event] = None,
    h5group: Optional[h5py.Group] = None,
) -> Optional[tuple[np.ndarray, np.ndarray, TimeSync]]:
    # dummy event
    evt_stop = evt_stop or threading.Event()

    ############################################################################
    # set up readout

    if ro.frequency_mhz_set != config.readout_frequency:
        freq_set = config.readout_frequency
        config.readout_frequency = ro.set_system_clock(freq_set)
        print(f'freq: {config.readout_frequency:0.1f} / {freq_set:0.1f} MHz')

    ############################################################################
    # configure readout & chip

    ro.set_threshold_voltage(config.voltage_threshold)
    ro.set_baseline_voltage(config.voltage_baseline)

    ro.sm_exec(prog_dac_config(config.dac_cfg.generate()))

    time.sleep(0.025)

    setup = ro.setup
    rows=list(int(x) for x in config.rows)

    ############################################################################
    # prepare statemachine

    pulse_cycles = math.ceil(config.pulse_ns * ro.frequency_mhz / 1000)
    frame_cycles = int(ro.frequency_mhz * config.frame_length_us)
    pause_cycles = int(ro.frequency_mhz * config.pause_length_us)

    if rows:
        prog_init, prog_readout = prog_read_frames(
            frame_cycles=frame_cycles,
            pulse_cycles=pulse_cycles,
            pause_cycles=pause_cycles,
            reset_counters=config.reset_counters,
            setup=setup,
            frequency=config.readout_frequency,
            # convert rows from np.int64
            rows=rows,
            read_adders=config.read_adders,
        )
    else:
        prog_init, prog_readout = prog_read_adders(
            frame_cycles=frame_cycles,
            pulse_cycles=pulse_cycles,
            pause_cycles=pause_cycles,
            frequency=config.readout_frequency,
            setup=setup,
        )
    prog_init.append(Finish())
    prog_readout.append(Finish())

    offset_init = ro.sm_write(prog_readout)
    ro.sm_write(prog_init, offset_init)

    ############################################################################
    # calculations

    numrows = len(config.rows)
    # raw frame duration
    duration_frame = 1e-6 * (config.frame_length_us + config.pause_length_us)
    # time for readout (estimated)
    readout_bits = setup.pixel_columns * (numrows + 1) * setup.chip.bits_adder
    duration_frame += 1e-6 * readout_bits / ro.frequency_mhz
    # number of frames per readout packet
    frames_per_run = min(
        # limit frames per run to 1 << 16 (hardware)
        (1 << 16) - 2,
        # limit amount of data in a packet to 2 MB for less jitter
        (2 << 20) // readout_bits,
    )
    # number of runs needed to reach target frames
    num_runs = math.ceil(config.num_frames / frames_per_run)
    # store real value in config
    config.frames_per_run = frames_per_run

    duration_run = frames_per_run * duration_frame
    duration_total = num_runs * duration_run

    print(f'{duration_run=:0.2f}s {frames_per_run=} {num_runs=}')

    timeout_run = 3.0 + 2.0 * duration_run
    timeout_total = 5.0 + 2.0 * duration_total

    if progress is not None:
        prog, task = progress
        if config.num_frames > 0:
            prog.start_task(task)
            prog.update(task, total=frames_per_run * num_runs)
        else:
            prog.update(task, total=1)

    ############################################################################
    # set up h5 storage

    time_sync = ro.get_synchronization()

    h5data = dset_times = None
    if h5group is not None:
        dset_frames = h5group.create_dataset(
            'frames',
            dtype=np.uint32,
            shape=(frames_per_run, numrows, setup.pixel_columns),
            maxshape=(None, numrows, setup.pixel_columns),
            chunks=(frames_per_run, numrows, setup.pixel_columns),
            compression='gzip',
        ) if numrows > 0 else None
        dset_adders = h5group.create_dataset(
            'adders',
            dtype=np.uint32,
            shape=(frames_per_run, setup.pixel_columns),
            maxshape=(None, setup.pixel_columns),
            chunks=(frames_per_run, setup.pixel_columns),
            compression='gzip',
        ) if config.read_adders else None
        dset_times = h5group.create_dataset(
            'times',
            dtype=np.uint32,
            # will get expanded later, zero size is not allowed
            shape=(frames_per_run,),
            maxshape=(None,),
            chunks=(frames_per_run,),
            compression='gzip',
        )
        h5data = (
            dset_frames,
            dset_adders,
            dset_times,
        )
        import json
        dset_times.attrs['sync'] = json.dumps(time_sync.asdict())

    ############################################################################
    # process data

    response_queue = queue.Queue()
    fastreadout.orphan_response_queue = response_queue
    adders = []
    frames = []
    timestamps = []

    t_decode = threading.Thread(
        target=_decode_responses,
        daemon=True,
        kwargs={
            'response_queue': response_queue,
            'frames': frames,
            'adders': adders,
            'timestamps': timestamps,
            'timeout': timeout_run,
            'setup': setup,
            'reset_counters': config.reset_counters,
            'evt_stop': evt_stop,
            'num_runs': num_runs if config.num_frames > 0 else None,
            'read_adders': config.read_adders,
            'num_rows': numrows,
            'callback': callback,
            'progress': progress,
            'h5data': h5data,
        },
    )
    t_decode.start()

    ############################################################################
    # run init program
    ro.sm_start(offset=offset_init)
    # start measurement
    ro.sm_start(
        frames_per_run,
        packets=num_runs if config.num_frames > 0 else 0,
    )

    if config.num_frames > 0:
        t_timeout = time.monotonic() + timeout_total
    else:
        t_timeout = float('+inf')
    # wait until idle
    while ro.get_sm_status().active:
        # stop event (SIGINT)
        if evt_stop.is_set():
            for _ in range(3):
                ro.sm_soft_abort()
            print('[yellow]sent soft abort signal')
            try:
                ro.wait_sm_idle(timeout_run)
            except TimeoutError:
                print('[red]sending hard abort signal')
                ro.sm_abort()
                ro.wait_sm_idle(timeout_run)
            break
        # timeout
        if time.monotonic() > t_timeout:
            ro.sm_abort()
            raise TimeoutError('statemachine not idle')
        # no need to react quickly here
        time.sleep(0.05)

    print('[yellow]waiting for decode thread to join')
    t_decode.join(10*timeout_run)
    if t_decode.is_alive():
        print('[red] decode thread still running')
        raise RuntimeError()
    print('[yellow]decode thread joined')

    ############################################################################

    # do not return any data when it was written to h5 file directly
    if h5group is not None:
        assert dset_times is not None
        if config.num_frames < 0:
            print('total frames: ', dset_times.len())
        config.num_frames = dset_times.len()
        return

    frames = np.concatenate(frames)
    timestamps = np.concatenate(timestamps)

    if config.num_frames < 0:
        print('total frames: ', len(timestamps))
    config.num_frames = len(timestamps)

    return frames, timestamps, time_sync
