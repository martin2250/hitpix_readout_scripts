from hitpix import HitPixColumnConfig, ReadoutPins, HitPixSetup
from readout.instructions import *
from readout.sm_prog import prog_shift_dense


def prog_injections_variable(
    num_injections: int,
    pulse_cycles: int,
    setup: HitPixSetup,
    rows: list[int],
    simultaneous_injections: int,
    frequency: float,
) -> list[Instruction]:
    # parameters
    chip = setup.chip
    for row in rows:
        assert row in range(chip.rows)
    assert setup.chip_rows == setup.chip_columns == 1
    assert (setup.pixel_columns % simultaneous_injections) == 0
    injection_steps = setup.pixel_columns // simultaneous_injections
    # column config
    def _get_injection_cc(
        inject_row: int,
        injection_step: int,
    ) -> HitPixColumnConfig:
        mask = 0
        for i in range(simultaneous_injections):
            bit = i * injection_steps + injection_step
            mask |= 1 << bit
        # readout inactive (col 24)
        return HitPixColumnConfig(
            inject_row=(1 << inject_row),
            inject_col=mask,
            ampout_col=0,
            rowaddr=-1,
        )
    # default configuration
    cfg_int = SetCfg(
        shift_rx_invert=True,
        shift_tx_invert=True,
        shift_toggle=True,
        shift_select_dac=False,
        shift_word_len=2 * chip.bits_adder,
        shift_clk_div=0,
        shift_sample_latency=setup.get_readout_latency(0, frequency),
    )
    pins = SetPins(0)
    prog = []
    cfg_col_prep = _get_injection_cc(0, 0)
    prog.extend([
        cfg_int,
        pins,
        Sleep(pulse_cycles),
        Reset(True, True),
        *prog_shift_dense(setup.encode_column_config(cfg_col_prep), False),
        Sleep(pulse_cycles),
        pins.set_pin(ReadoutPins.ro_ldconfig, True),
        Sleep(pulse_cycles),
        pins,
    ])
    for injection_step in range(injection_steps):
        for row_idx in range(len(rows)):
            # variables
            row_idx_next = (row_idx + 1) % len(rows)
            row = rows[row_idx]
            row_next = rows[row_idx_next]
            # last row -> prepare for next injection step
            if row_idx_next == 0:
                injection_step_next = (injection_step + 1) % injection_steps
            else:
                injection_step_next = injection_step
            # read out current row after injections
            cfg_col_readout = HitPixColumnConfig(0, 0, 0, row)
            # inject into next row in next loop iteration
            cfg_col_inj_next = _get_injection_cc(row_next, injection_step_next)
            prog.extend([
                # reset counter
                pins.set_pin(ReadoutPins.ro_rescnt, True),
                Sleep(pulse_cycles),
                pins,
                Sleep(pulse_cycles),
                # set frame high and inject
                pins.set_pin(ReadoutPins.ro_frame, True),
                Sleep(pulse_cycles),
                Inject(num_injections),
                Sleep(pulse_cycles),
                pins,
                Sleep(pulse_cycles),
                # shift in column config for readout
                Reset(True, True),
                *prog_shift_dense(setup.encode_column_config(cfg_col_readout), False),
                Sleep(pulse_cycles),
                pins.set_pin(ReadoutPins.ro_ldconfig, True),
                Sleep(pulse_cycles),
                pins,
                Sleep(pulse_cycles),
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
                # add time to make readout more consistent
                GetTime(),
                # read out data while shifting in configuration for the next round of injections
                Reset(True, True),
                *prog_shift_dense(setup.encode_column_config(cfg_col_inj_next), True),
                Sleep(pulse_cycles),
                pins.set_pin(ReadoutPins.ro_ldconfig, True),
                Sleep(pulse_cycles),
                pins,
                Sleep(pulse_cycles),
            ])
    prog.append(Finish())
    return prog
