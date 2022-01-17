import bitarray
import bitarray.util
import numpy as np
from hitpix import HitPixColumnConfig, ReadoutPins, HitPixSetup
from readout.instructions import *
from readout.sm_prog import prog_shift_dense




def prog_injections_half(num_injections: int, shift_clk_div: int, pulse_cycles: int, setup: HitPixSetup) -> list[Instruction]:
    assert setup.chip_rows == 1
    chip = setup.chip
    def _get_cfg_col_injection(inject_row: int, readout_row: int, odd_pixels: bool) -> HitPixColumnConfig:
        mask = int(('01' if odd_pixels else '10') * (chip.columns // 2), 2)
        # readout inactive (col 24)
        return HitPixColumnConfig(
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
        shift_word_len=2 * chip.bits_adder,
        shift_clk_div=shift_clk_div,
        pins=0,
    )
    prog = []
    cfg_col_prep = _get_cfg_col_injection(0, -1, False)
    prog.extend([
        cfg_int,
        Sleep(pulse_cycles),
        Reset(True, True),
        *prog_shift_dense(setup.encode_column_config(cfg_col_prep), False),
        Sleep(pulse_cycles),
        cfg_int.set_pin(ReadoutPins.ro_ldconfig, True),
        Sleep(pulse_cycles),
        cfg_int,
    ])
    for row in range(2*chip.rows):
        # read out current row after injections
        cfg_col_readout = HitPixColumnConfig(0, 0, 0, row % chip.rows)
        # inject into next row in next loop iteration
        row_inj_next = (row + 1) % chip.rows
        odd_pixels_next = (row + 1) >= chip.rows
        cfg_col_inj_next = _get_cfg_col_injection(
            row_inj_next, -1, odd_pixels_next)
        prog.extend([
            # reset counter
            cfg_int.set_pin(ReadoutPins.ro_rescnt, True),
            Sleep(pulse_cycles),
            cfg_int,
            Sleep(pulse_cycles),
            # set frame high and inject
            cfg_int.set_pin(ReadoutPins.ro_frame, True),
            Sleep(pulse_cycles),
            Inject(num_injections),
            Sleep(pulse_cycles),
            cfg_int,
            Sleep(pulse_cycles),
            # shift in column config for readout
            Reset(True, True),
            *prog_shift_dense(setup.encode_column_config(cfg_col_readout), False),
            Sleep(pulse_cycles),
            cfg_int.set_pin(ReadoutPins.ro_ldconfig, True),
            Sleep(pulse_cycles),
            cfg_int,
            Sleep(pulse_cycles),
            # load count into column register
            cfg_int.set_pin(ReadoutPins.ro_ldcnt, True),
            Sleep(pulse_cycles),
            cfg_int,
            Sleep(pulse_cycles),
            cfg_int.set_pin(ReadoutPins.ro_penable, True),
            Sleep(pulse_cycles),
            ShiftOut(1, False),
            Sleep(pulse_cycles),
            cfg_int,
            Sleep(pulse_cycles),
            # add time to make readout more consistent
            GetTime(),
            # read out data while shifting in configuration for the next round of injections
            Reset(True, True),
            *prog_shift_dense(setup.encode_column_config(cfg_col_inj_next), True),
            Sleep(pulse_cycles),
            cfg_int.set_pin(ReadoutPins.ro_ldconfig, True),
            Sleep(pulse_cycles),
            cfg_int,
            Sleep(pulse_cycles),
        ])
    prog.append(Finish())
    return prog

def prog_injections_full(num_injections: int, shift_clk_div: int, pulse_cycles: int, setup: HitPixSetup) -> list[Instruction]:
    assert setup.chip_rows == 1
    chip = setup.chip
    cfg_int = SetCfg(
        shift_rx_invert=True,
        shift_tx_invert=True,
        shift_toggle=True,
        shift_select_dac=False,
        shift_word_len=2 * chip.bits_adder,
        shift_clk_div=shift_clk_div,
        pins=0,
    )
    prog = []
    # prepare chip for first injections
    cfg_col_prep = HitPixColumnConfig(
        inject_row=(1 << 0),
        inject_col=(1 << chip.columns) - 1,
        ampout_col=0,
        rowaddr=-1,
    )
    prog.extend([
        cfg_int,
        Sleep(pulse_cycles),
        Reset(True, True),
        *prog_shift_dense(setup.encode_column_config(cfg_col_prep), False),
        Sleep(pulse_cycles),
        cfg_int.set_pin(ReadoutPins.ro_ldconfig, True),
        Sleep(pulse_cycles),
        cfg_int,
    ])
    for row in range(chip.rows):
        # read out current row after injections
        cfg_col_readout = HitPixColumnConfig(0, 0, 0, row % chip.rows)
        # inject into next row in next loop iteration
        row_inj_next = (row + 1) % chip.rows
        cfg_col_inj_next = HitPixColumnConfig(
            inject_row=(1 << row_inj_next),
            inject_col=(1 << chip.columns) - 1,
            ampout_col=0,
            rowaddr=-1,
        )
        prog.extend([
            # reset counter
            cfg_int.set_pin(ReadoutPins.ro_rescnt, True),
            Sleep(pulse_cycles),
            cfg_int,
            Sleep(pulse_cycles),
            # set frame high and inject
            cfg_int.set_pin(ReadoutPins.ro_frame, True),
            Sleep(pulse_cycles),
            Inject(num_injections),
            Sleep(pulse_cycles),
            cfg_int,
            Sleep(pulse_cycles),
            # shift in column config for readout
            Reset(True, True),
            *prog_shift_dense(setup.encode_column_config(cfg_col_readout), False),
            Sleep(pulse_cycles),
            cfg_int.set_pin(ReadoutPins.ro_ldconfig, True),
            Sleep(pulse_cycles),
            cfg_int,
            Sleep(pulse_cycles),
            # load count into column register
            cfg_int.set_pin(ReadoutPins.ro_ldcnt, True),
            Sleep(pulse_cycles),
            cfg_int,
            Sleep(pulse_cycles),
            cfg_int.set_pin(ReadoutPins.ro_penable, True),
            Sleep(pulse_cycles),
            ShiftOut(1, False),
            Sleep(pulse_cycles),
            cfg_int,
            Sleep(pulse_cycles),
            # add time to make readout more consistent
            GetTime(),
            # read out data while shifting in configuration for the next round of injections
            Reset(True, True),
            *prog_shift_dense(setup.encode_column_config(cfg_col_inj_next), True),
            Sleep(pulse_cycles),
            cfg_int.set_pin(ReadoutPins.ro_ldconfig, True),
            Sleep(pulse_cycles),
            cfg_int,
            Sleep(pulse_cycles),
        ])
    prog.append(Finish())
    return prog
