from hitpix import HitPixColumnConfig, HitPixSetup, ReadoutPins
from readout.instructions import *
from readout.sm_prog import prog_shift_dense, prog_sleep, sample_latency


def prog_read_frames(
    frame_cycles: int,
    pulse_cycles: int,
    shift_clk_div: int,
    pause_cycles: int,
    reset_counters: bool,
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
        shift_sample_latency=sample_latency[shift_clk_div],
    )
    pins = SetPins(0)

    # init
    col_cfg_init = HitPixColumnConfig(0, 0, 0, -1)
    prog_init = [
        Reset(True, True),
        cfg_int,
        pins,
        *prog_shift_dense(setup.encode_column_config(col_cfg_init), False),
        pins.set_pin(ReadoutPins.ro_ldconfig, True),
        Sleep(pulse_cycles),
        pins,
    ]

    # readout
    prog: list[Instruction] = [
        cfg_int
    ]
    # reset counter every frame?
    if reset_counters:
        prog.extend([
            # reset counters
            pins.set_pin(ReadoutPins.ro_rescnt, True),
            Sleep(pulse_cycles),
            pins,
            Sleep(pulse_cycles),
        ])

    prog.extend([
        # wait (empty if pause_cycles = 0)
        *prog_sleep(pause_cycles),
        # take data
        pins.set_pin(ReadoutPins.ro_frame, True),
        *prog_sleep(frame_cycles),
        pins,
        Sleep(pulse_cycles),
    ])

    for row in range(chip.rows + 1):
        col_cfg = HitPixColumnConfig(0, 0, 0, row)
        # add time to make readout more consistent
        if row > 0:
            prog.append(GetTime())
        prog.extend([
            Sleep(pulse_cycles),
            Reset(True, True),
            *prog_shift_dense(setup.encode_column_config(col_cfg), row > 0),
            Sleep(pulse_cycles),
            pins.set_pin(ReadoutPins.ro_ldconfig, True),
            Sleep(pulse_cycles),
            pins,
            Sleep(pulse_cycles),
        ])
        if row == chip.rows:
            break
        prog.extend([
            # load count into column register
            pins.set_pin(ReadoutPins.ro_ldcnt, True),
            Sleep(pulse_cycles),
            pins,
            Sleep(pulse_cycles),
            pins.set_pin(ReadoutPins.ro_penable, True),
            Sleep(pulse_cycles),
            ShiftOut(1, False),
            Sleep(pulse_cycles),
            pins,
            Sleep(pulse_cycles),
        ])

    return prog_init, prog
