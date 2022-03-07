import math
import queue
import threading
import time
from typing import Optional

import h5py
import numpy as np
import tqdm
from hitpix import HitPixSetup
from hitpix.readout import HitPixReadout
from readout.fast_readout import FastReadout
from readout.instructions import Finish
from readout.sm_prog import decode_column_packets, prog_dac_config
from util.time_sync import TimeSync

from .io import FrameConfig
from .sm_prog import prog_read_frames


def _decode_responses(
    response_queue: queue.Queue[bytes],
    frames: list[np.ndarray],
    timestamps: list[np.ndarray],
    timeout: float,
    setup: HitPixSetup,
    reset_counters: bool,
    num_runs: Optional[int] = None,
    callback=None,
    progress: Optional[tqdm.tqdm] = None,
    h5data: Optional[tuple[h5py.Dataset, h5py.Dataset]] = None,
):
    ctr_max = 1 << setup.chip.bits_counter
    hits_last = None
    # total number of frames received so far
    frames_total = 0

    assert num_runs is not None
    for _ in range(num_runs):
        data = response_queue.get(True, timeout)

        # decode hits
        block_timestamps, block_frames = decode_column_packets(
            data,
            setup.pixel_columns,
            setup.chip.bits_adder,
            setup.chip.bits_counter,
        )
        block_frames = block_frames.reshape(
            -1,
            setup.pixel_rows,
            setup.pixel_columns,
        )
        block_numframes = block_frames.shape[0]

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
            dset_frames, dset_times = h5data
            size_new = frames_total+block_numframes
            slice_new = slice(frames_total, None)
            # store frames
            dset_frames.resize(size_new, 0)
            dset_frames[slice_new] = block_frames
            # store timestamps
            dset_times.resize(size_new, 0)
            dset_times[slice_new] = block_timestamps[::setup.pixel_rows]
        else:
            frames.append(block_frames)
            # only store timestamp of first row of each frame
            timestamps.append(block_timestamps[::setup.pixel_rows])

        frames_total += block_numframes
        if progress is not None:
            progress.update(block_numframes)


def read_frames(
    ro: HitPixReadout,
    fastreadout: FastReadout,
    config: FrameConfig,
    progress: Optional[tqdm.tqdm] = None,
    callback=None,
    h5group: Optional[h5py.Group] = None,
) -> Optional[tuple[np.ndarray, np.ndarray, TimeSync]]:
    ############################################################################
    # set up readout

    if ro.frequency_mhz_set != config.readout_frequency:
        print(f'setting frequency to {config.readout_frequency=}')
        config.readout_frequency = ro.set_system_clock(
            config.readout_frequency)
        print(f'actual: {config.readout_frequency}')

    ############################################################################
    # configure readout & chip

    ro.set_threshold_voltage(config.voltage_threshold)
    ro.set_baseline_voltage(config.voltage_baseline)

    ro.sm_exec(prog_dac_config(config.dac_cfg.generate()))

    time.sleep(0.025)

    setup = ro.setup

    ############################################################################
    # prepare statemachine

    pulse_cycles = int(max(1, config.pulse_ns * ro.frequency_mhz / 1000))

    prog_init, prog_readout = prog_read_frames(
        frame_cycles=int(ro.frequency_mhz * config.frame_length_us),
        pulse_cycles=pulse_cycles,
        pause_cycles=int(ro.frequency_mhz * config.pause_length_us),
        reset_counters=config.reset_counters,
        setup=setup,
        frequency=config.readout_frequency,
    )
    prog_readout.append(Finish())

    ro.sm_exec(prog_init)
    ro.sm_write(prog_readout)

    ############################################################################
    # start measurement

    # raw frame duration
    duration_frame = 1e-6 * (config.frame_length_us + config.pause_length_us)
    # time for readout (estimated)
    readout_bits = setup.pixel_columns * setup.pixel_rows * setup.chip.bits_adder
    duration_frame += 1e-6 * readout_bits / ro.frequency_mhz
    # X ms per packet for good progress
    frames_per_run = math.ceil(500e-3 / duration_frame)
    # limit amount of data in a packet for less jitter
    if frames_per_run > 1024:
        frames_per_run = 1024
    num_runs = math.ceil(config.num_frames / frames_per_run)
    # store real values in config
    config.frames_per_run = frames_per_run
    config.num_frames = frames_per_run * num_runs

    duration_run = frames_per_run * duration_frame
    duration_total = num_runs * duration_run

    timeout_run = 1.0 + 1.5 * duration_run
    timeout_total = 5.0 + 1.5 * duration_total

    if progress is not None:
        progress.total = config.num_frames

    ############################################################################
    # set up h5 storage

    time_sync = ro.get_synchronization()

    h5data = None
    if h5group is not None:
        dset_frames = h5group.create_dataset(
            'frames',
            dtype=np.uint32,
            shape=(frames_per_run, setup.pixel_rows, setup.pixel_rows),
            maxshape=(None, setup.pixel_rows, setup.pixel_rows),
            chunks=(frames_per_run, setup.pixel_rows, setup.pixel_rows),
            compression='gzip',
        )
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
            dset_times,
        )
        import json
        dset_times.attrs['sync'] = json.dumps(time_sync.asdict())

    ############################################################################
    # process data

    response_queue = queue.Queue()
    fastreadout.orphan_response_queue = response_queue
    frames = []
    timestamps = []

    t_decode = threading.Thread(
        target=_decode_responses,
        daemon=True,
        kwargs={
            'response_queue': response_queue,
            'frames': frames,
            'timestamps': timestamps,
            'timeout': timeout_run,
            'setup': setup,
            'reset_counters': config.reset_counters,
            'num_runs': num_runs,
            'callback': callback,
            'progress': progress,
            'h5data': h5data,
        },
    )
    t_decode.start()

    ############################################################################

    # start measurement
    ro.sm_start(frames_per_run, packets=num_runs)
    ro.wait_sm_idle(timeout_total)

    t_decode.join(2*timeout_run)

    ############################################################################

    # do not return any data when it was written to h5 file directly
    if h5group is not None:
        return

    frames = np.concatenate(frames)
    timestamps = np.concatenate(timestamps)

    return frames, timestamps, time_sync
