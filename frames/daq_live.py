'''
this is supposed to be a continuous-daq frame capture, which may be interrupted anytime to change dac settings etc.
'''
import math
import queue
import threading
import time
from typing import Callable, Optional

import h5py
import numpy as np
from rich import print
from hitpix import HitPix2DacConfig, HitPixDacConfig, HitPixSetup
from hitpix.readout import HitPixReadout
from readout.fast_readout import FastReadout
from readout.instructions import Finish, count_cycles
from readout.sm_prog import decode_column_packets, prog_dac_config
from util.time_sync import TimeSync

from .io import FrameConfig
from .sm_prog import prog_read_adders, prog_read_adders_parallel, prog_read_frames, prog_read_frames_parallel

from dataclasses import dataclass


@dataclass
class FramesLiveSmConfig:
    pulse_ns: float
    frame_us: float
    pause_us: float
    

def live_write_statemachine(
    ro: HitPixReadout,
    config: FramesLiveSmConfig,
    rows: list[int],
    reset_counters: bool,
    dac_config: Optional[HitPix2DacConfig],
) -> float:
    '''returns time for single frame in us'''
    pulse_cycles = math.ceil(config.pulse_ns * ro.frequency_mhz / 1000)
    frame_cycles = int(ro.frequency_mhz * config.frame_us)
    pause_cycles = int(ro.frequency_mhz * config.pause_us)

    if rows:
        if ro.setup.parallel_readout:
            assert dac_config is not None
            prog_init, prog_readout = prog_read_frames_parallel(
                frame_cycles=frame_cycles,
                pulse_cycles=pulse_cycles,
                pause_cycles=pause_cycles,
                reset_counters=reset_counters,
                setup=ro.setup,
                frequency=ro.frequency_mhz,
                rows=rows,
                dac_config=dac_config,
            )
        else:
            prog_init, prog_readout = prog_read_frames(
                frame_cycles=frame_cycles,
                pulse_cycles=pulse_cycles,
                pause_cycles=pause_cycles,
                reset_counters=reset_counters,
                setup=ro.setup,
                frequency=ro.frequency_mhz,
                rows=rows,
                read_adders=False,
            )
    else:
        if ro.setup.parallel_readout:
            assert dac_config is not None
            prog_init, prog_readout = prog_read_adders_parallel(
                frame_cycles=frame_cycles,
                pulse_cycles=pulse_cycles,
                pause_cycles=pause_cycles,
                frequency=ro.frequency_mhz,
                setup=ro.setup,
                dac_config=dac_config,
            )
        else:
            prog_init, prog_readout = prog_read_adders(
                frame_cycles=frame_cycles,
                pulse_cycles=pulse_cycles,
                pause_cycles=pause_cycles,
                frequency=ro.frequency_mhz,
                setup=ro.setup,
            )
    prog_init.append(Finish())
    prog_readout.append(Finish())

    offset_init = ro.sm_write(prog_readout)
    ro.sm_write(prog_init, offset_init)
    ro.sm_start(offset=offset_init)
    ro.wait_sm_idle()

    return count_cycles(prog_readout) / ro.frequency_mhz

def live_decode_responses(
    response_queue: queue.Queue[bytes],
    setup: HitPixSetup,
    reset_counters: bool,
    num_rows: int, # 0 == adders
    callback: Callable[[np.ndarray], None],
):
    ctr_max = 1 << setup.chip.bits_counter
    adder_max = ctr_max * setup.chip.rows
    hits_last = None

    mask_adders = (1 << setup.chip.bits_adder) - 1
    mask_counters = (1 << setup.chip.bits_counter) - 1

    # data processing loop
    while True:
        try:
            data = response_queue.get(True)
            if (qs := response_queue.qsize()) > 10:
                print(f'[red] queue overflowing: {qs}')
            # decode hits
            _, block_frames = decode_column_packets(
                data,
                setup.decode_columns,
                setup.chip.bits_adder,
                None,
            )

            if num_rows == 0:
                block_adders = block_frames.reshape(
                    -1,
                    setup.pixel_columns,
                )
                block_adders = np.bitwise_and(block_adders, mask_adders)
                # counter counts down
                block_adders = (adder_max - block_adders) % adder_max
                # calculate quantile, hack for testbeam, TODO: make this pretty
                if True:
                    hits_per_frame = np.sum(block_adders, axis=1)
                    hits_95 = np.quantile(hits_per_frame, 0.95)
                    print(f'95th quantile hits/frame: {hits_95}')
                block_adders =  np.sum(block_adders, axis=0)
                callback(block_adders)
            else:
                block_frames = setup.reshape_data(block_frames, num_rows)
                block_frames = np.bitwise_and(block_frames, mask_counters)
                # counters not reset? apply diff between frames
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
                # calculate quantile, hack for testbeam, TODO: make this pretty
                if True:
                    hits_per_frame = np.sum(block_frames, axis=(1, 2))
                    hits_95 = np.quantile(hits_per_frame, 0.95)
                    print(f'95th quantile hits/frame: {hits_95}')
                block_frames =  np.sum(block_frames, axis=0)
                callback(block_frames)
        except Exception as e:
            print(f'[red] decoding exception {e}')
