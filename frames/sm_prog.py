from hitpix.hitpix1 import HitPix1ColumnConfig, HitPix1Pins
from readout.instructions import *
from readout.sm_prog import prog_shift_dense, prog_sleep


def prog_read_frames(frame_cycles: int, pulse_cycles: int = 10, shift_clk_div: int = 1, pause_cycles: int = 0) -> tuple[list[Instruction], list[Instruction]]:
    '''returns programs (init and readout)'''
    cfg_int = SetCfg(
        shift_rx_invert=True,
        shift_tx_invert=True,
        shift_toggle=True,
        shift_select_dac=False,
        shift_word_len=2 * 13,
        shift_clk_div=shift_clk_div,
        pins=0,
    )

    # init
    col_cfg_init = HitPix1ColumnConfig(0, 0, 0, 24)
    prog_init = [
        Reset(True, True),
        *prog_shift_dense(col_cfg_init.generate(), False),
        cfg_int.set_pin(HitPix1Pins.ro_ldconfig, True),
        Sleep(pulse_cycles),
        cfg_int,
    ]

    # readout
    prog = [
        # wait (empty if pause_cycles = 0)
        *prog_sleep(pause_cycles),
        # reset counters
        cfg_int.set_pin(HitPix1Pins.ro_rescnt, True),
        Sleep(pulse_cycles),
        cfg_int,
        # take data
        cfg_int.set_pin(HitPix1Pins.ro_frame, True),
        *prog_sleep(frame_cycles),
        cfg_int,
    ]
    for row in range(25):
        col_cfg = HitPix1ColumnConfig(0, 0, 0, row)
        # add time to make readout more consistent
        if row > 0:
            prog.append(GetTime())
        prog.extend([
            Sleep(pulse_cycles),
            Reset(True, True),
            *prog_shift_dense(col_cfg.generate(), row > 0),
            Sleep(pulse_cycles),
            cfg_int.set_pin(HitPix1Pins.ro_ldconfig, True),
            Sleep(pulse_cycles),
            cfg_int,
            Sleep(pulse_cycles),
        ])
        if row == 24:
            break
        prog.extend([
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
        ])

    return prog_init, prog
