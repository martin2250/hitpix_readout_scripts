from hitpix import HitPixColumnConfig, HitPixSetup, ReadoutPins
from readout.instructions import *
from readout.sm_prog import prog_shift_dense


def prog_laser_inject(
    injections_per_round: int,
    pulse_cycles: int,
    shift_clk_div: int,
    setup: HitPixSetup,
) -> tuple[list[Instruction], list[Instruction]]:
    '''returns programs (init and readout)'''
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

    # init
    col_cfg_init = HitPixColumnConfig(0, 0, 0, -1)
    prog_init = [
        Reset(True, True),
        *prog_shift_dense(setup.encode_column_config(col_cfg_init), False),
        Sleep(3),
        cfg_int.set_pin(ReadoutPins.ro_ldconfig, True),
        Sleep(pulse_cycles),
        cfg_int,
    ]

    # readout
    prog = [
        # reset counters
        cfg_int.set_pin(ReadoutPins.ro_rescnt, True),
        Sleep(pulse_cycles),
        cfg_int,
        # take data
        cfg_int.set_pin(ReadoutPins.ro_frame, True),
        Sleep(pulse_cycles),
        Inject(injections_per_round),
        Sleep(pulse_cycles),
        cfg_int,
    ]
    for row in range(chip.rows + 1):
        col_cfg = HitPixColumnConfig(0, 0, 0, row)
        # add time to make readout more consistent
        if row > 0:
            prog.append(GetTime())
        prog.extend([
            Sleep(pulse_cycles),
            Reset(True, True),
            *prog_shift_dense(setup.encode_column_config(col_cfg), row > 0),
            Sleep(3),
            Sleep(pulse_cycles),
            cfg_int.set_pin(ReadoutPins.ro_ldconfig, True),
            Sleep(pulse_cycles),
            cfg_int,
            Sleep(pulse_cycles),
        ])
        if row == chip.rows:
            break
        prog.extend([
            # load count into column register
            cfg_int.set_pin(ReadoutPins.ro_ldcnt, True),
            Sleep(pulse_cycles),
            cfg_int,
            Sleep(pulse_cycles),
            cfg_int.set_pin(ReadoutPins.ro_penable, True),
            Sleep(pulse_cycles),
            ShiftOut(1, False),
            Sleep(pulse_cycles + 3),
            cfg_int,
            Sleep(pulse_cycles),
        ])

    return prog_init, prog
