import bitarray
import bitarray.util
import numpy as np
from hitpix.hitpix1 import HitPix1ColumnConfig, HitPix1Pins
from readout.instructions import *
from readout.sm_prog import prog_shift_dense




def prog_injections_half(num_injections: int, shift_clk_div: int = 1, pulse_cycles: int = 10) -> list[Instruction]:
    def _get_cfg_col_injection(inject_row: int, readout_row: int, odd_pixels: bool) -> HitPix1ColumnConfig:
        mask = int(('01' if odd_pixels else '10') * 12, 2)
        # readout inactive (col 24)
        return HitPix1ColumnConfig(
            inject_row=(1 << inject_row),
            inject_col=mask,
            ampout_col=0,
            rowaddr=readout_row,
        )
    cfg_int = SetCfg(
        shift_rx_invert=True,
        shift_tx_invert=True,
        shift_toggle=True,
        shift_select_dac=False,
        shift_word_len=2 * 13,
        shift_clk_div=shift_clk_div,
        pins=0,
    )
    prog = []
    cfg_col_prep = _get_cfg_col_injection(0, 24, False)
    prog.extend([
        cfg_int,
        Sleep(pulse_cycles),
        Reset(True, True),
        *prog_shift_dense(cfg_col_prep.generate(), False),
        Sleep(pulse_cycles),
        cfg_int.set_pin(HitPix1Pins.ro_ldconfig, True),
        Sleep(pulse_cycles),
        cfg_int,
    ])
    for row in range(2*24):
        # read out current row after injections
        cfg_col_readout = HitPix1ColumnConfig(0, 0, 0, row % 24)
        # inject into next row in next loop iteration
        row_inj_next = (row + 1) % 24
        odd_pixels_next = (row + 1) >= 24
        cfg_col_inj_next = _get_cfg_col_injection(
            row_inj_next, 24, odd_pixels_next)
        prog.extend([
            # reset counter
            cfg_int.set_pin(HitPix1Pins.ro_rescnt, True),
            Sleep(pulse_cycles),
            cfg_int,
            Sleep(pulse_cycles),
            # set frame high and inject
            cfg_int.set_pin(HitPix1Pins.ro_frame, True),
            Sleep(pulse_cycles),
            Inject(num_injections),
            Sleep(pulse_cycles),
            cfg_int,
            Sleep(pulse_cycles),
            # shift in column config for readout
            Reset(True, True),
            *prog_shift_dense(cfg_col_readout.generate(), False),
            Sleep(pulse_cycles),
            cfg_int.set_pin(HitPix1Pins.ro_ldconfig, True),
            Sleep(pulse_cycles),
            cfg_int,
            Sleep(pulse_cycles),
            # load count into column register
            cfg_int.set_pin(HitPix1Pins.ro_ldcnt, True),
            Sleep(pulse_cycles),
            cfg_int,
            Sleep(pulse_cycles),
            cfg_int.set_pin(HitPix1Pins.ro_penable, True),
            Sleep(pulse_cycles),
            ShiftOut(1, False),
            Sleep(pulse_cycles),
            cfg_int,
            Sleep(pulse_cycles),
            # add time to make readout more consistent
            GetTime(),
            # read out data while shifting in configuration for the next round of injections
            Reset(True, True),
            *prog_shift_dense(cfg_col_inj_next.generate(), True),
            Sleep(pulse_cycles),
            cfg_int.set_pin(HitPix1Pins.ro_ldconfig, True),
            Sleep(pulse_cycles),
            cfg_int,
            Sleep(pulse_cycles),
        ])
    prog.append(Finish())
    return prog

def prog_injections_full(num_injections: int, shift_clk_div: int = 1, pulse_cycles: int = 10) -> list[Instruction]:
    cfg_int = SetCfg(
        shift_rx_invert=True,
        shift_tx_invert=True,
        shift_toggle=True,
        shift_select_dac=False,
        shift_word_len=2 * 13,
        shift_clk_div=shift_clk_div,
        pins=0,
    )
    prog = []
    # prepare chip for first injections
    cfg_col_prep = HitPix1ColumnConfig(
        inject_row=(1 << 0),
        inject_col=(1 << 24) - 1,
        ampout_col=0,
        rowaddr=24,
    )
    prog.extend([
        cfg_int,
        Sleep(pulse_cycles),
        Reset(True, True),
        *prog_shift_dense(cfg_col_prep.generate(), False),
        Sleep(pulse_cycles),
        cfg_int.set_pin(HitPix1Pins.ro_ldconfig, True),
        Sleep(pulse_cycles),
        cfg_int,
    ])
    for row in range(24):
        # read out current row after injections
        cfg_col_readout = HitPix1ColumnConfig(0, 0, 0, row % 24)
        # inject into next row in next loop iteration
        row_inj_next = (row + 1) % 24
        cfg_col_inj_next = HitPix1ColumnConfig(
            inject_row=(1 << row_inj_next),
            inject_col=(1 << 24) - 1,
            ampout_col=0,
            rowaddr=24,
        )
        prog.extend([
            # reset counter
            cfg_int.set_pin(HitPix1Pins.ro_rescnt, True),
            Sleep(pulse_cycles),
            cfg_int,
            Sleep(pulse_cycles),
            # set frame high and inject
            cfg_int.set_pin(HitPix1Pins.ro_frame, True),
            Sleep(pulse_cycles),
            Inject(num_injections),
            Sleep(pulse_cycles),
            cfg_int,
            Sleep(pulse_cycles),
            # shift in column config for readout
            Reset(True, True),
            *prog_shift_dense(cfg_col_readout.generate(), False),
            Sleep(pulse_cycles),
            cfg_int.set_pin(HitPix1Pins.ro_ldconfig, True),
            Sleep(pulse_cycles),
            cfg_int,
            Sleep(pulse_cycles),
            # load count into column register
            cfg_int.set_pin(HitPix1Pins.ro_ldcnt, True),
            Sleep(pulse_cycles),
            cfg_int,
            Sleep(pulse_cycles),
            cfg_int.set_pin(HitPix1Pins.ro_penable, True),
            Sleep(pulse_cycles),
            ShiftOut(1, False),
            Sleep(pulse_cycles),
            cfg_int,
            Sleep(pulse_cycles),
            # add time to make readout more consistent
            GetTime(),
            # read out data while shifting in configuration for the next round of injections
            Reset(True, True),
            *prog_shift_dense(cfg_col_inj_next.generate(), True),
            Sleep(pulse_cycles),
            cfg_int.set_pin(HitPix1Pins.ro_ldconfig, True),
            Sleep(pulse_cycles),
            cfg_int,
            Sleep(pulse_cycles),
        ])
    prog.append(Finish())
    return prog
