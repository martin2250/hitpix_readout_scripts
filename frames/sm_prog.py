import dataclasses
from typing import Iterable

from bitarray import bitarray
from hitpix import HitPix2DacConfig, HitPixColumnConfig, HitPixSetup, ReadoutPins
from readout.instructions import *
from readout.sm_prog import prog_shift_dense, prog_sleep

try:
    from rich import print
except ImportError:
    pass


def prog_read_frames(
    frame_cycles: int,
    pulse_cycles: int,
    pause_cycles: int,
    reset_counters: bool,
    frequency: float,
    setup: HitPixSetup,
    rows: Iterable[int],
    read_adders: bool,
) -> tuple[list[Instruction], list[Instruction]]:
    '''returns programs (init and readout)'''
    chip = setup.chip

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
    pulse_sleep = prog_sleep(pulse_cycles - 1, cfg_int)

    # init
    col_cfg_init = HitPixColumnConfig(0, 0, 0, -1)
    prog_init = [
        Reset(True, True),
        cfg_int,
        pins,
        *prog_shift_dense(setup.encode_column_config(col_cfg_init), False),
        Sleep(3),
        *pulse_sleep,
        pins.set_pin(ReadoutPins.ro_ldconfig, True),
        *pulse_sleep,
        pins,
    ]

    # readout
    rows_curr = [-1] + list(rows)
    rows_next = list(rows) + [-1]

    prog: list[Instruction] = [
        cfg_int
    ]
    # reset counter every frame?
    if reset_counters:
        prog.extend([
            # reset counters
            pins.set_pin(ReadoutPins.ro_rescnt, True),
            *pulse_sleep,
            pins,
            *pulse_sleep,
        ])

    prog.extend([
        # wait (empty if pause_cycles = 0)
        *prog_sleep(pause_cycles),
        # take data
        pins.set_pin(ReadoutPins.ro_frame, True),
        *prog_sleep(frame_cycles),
        pins,
        *pulse_sleep,
    ])

    for row_curr, row_next in zip(rows_curr, rows_next):
        shift_col_cfg = HitPixColumnConfig(0, 0, 0, row_next)
        shift_shiftout = False
        # read out data when the currently selected row is not -1
        # when reading adders, use row -1 -> no extra operation
        if read_adders or row_curr != -1:
            shift_shiftout = True
            pins_penable = pins.set_pin(ReadoutPins.ro_penable, True)
            # reading adders? use psel
            if row_curr == -1:
                pins_penable = pins_penable.set_pin(ReadoutPins.ro_psel, True)
            # load data into shift register
            prog.extend([
                # add time to every row to make readout more consistent
                GetTime(),
                # load count into column register
                pins.set_pin(ReadoutPins.ro_ldcnt, True),
                *pulse_sleep,
                pins,
                *pulse_sleep,
                # shift one bit
                pins_penable,
                *pulse_sleep,
                ShiftOut(1, False),
                *prog_sleep(pulse_cycles + 3, cfg_int),
                pins,
                *pulse_sleep,
            ])
        # shift in + out data
        prog.extend([
            Reset(True, True),
            *prog_shift_dense(setup.encode_column_config(shift_col_cfg), shift_shiftout),
            Sleep(3),
            *pulse_sleep,
            pins.set_pin(ReadoutPins.ro_ldconfig, True),
            *pulse_sleep,
            pins,
            *pulse_sleep,
        ])
    # make sure that all data is sent
    # (otherwise readout packets could end up short with small pulse_cycles)
    prog.append(Sleep(4))
    return prog_init, prog


def prog_read_adders(
    frame_cycles: int,
    pulse_cycles: int,
    pause_cycles: int,
    frequency: float,
    setup: HitPixSetup,
) -> tuple[list[Instruction], list[Instruction]]:
    '''returns programs (init and readout)'''
    chip = setup.chip

    cfg_int = SetCfg(
        shift_rx_invert=True,
        shift_tx_invert=True,
        shift_toggle=True,
        shift_select_dac=False,
        shift_word_len=2 * chip.bits_adder,
        shift_clk_div=0,
        shift_sample_latency=setup.get_readout_latency(0, frequency),
    )
    pins = SetPins(0).set_pin(ReadoutPins.ro_psel, True)
    pins_frame = pins.set_pin(ReadoutPins.ro_frame, True)
    pulse_sleep = prog_sleep(pulse_cycles - 1, cfg_int)

    # init
    col_cfg = HitPixColumnConfig(0, 0, 0, -1)
    prog_init = [
        cfg_int,
        pins,
        Reset(True, True),
        *prog_shift_dense(setup.encode_column_config(col_cfg), False),
        Sleep(3),
        pins.set_pin(ReadoutPins.ro_ldconfig, True),
        *pulse_sleep,
        pins,
        # reset counters
        pins.set_pin(ReadoutPins.ro_rescnt, True),
        *pulse_sleep,
        pins,
        *pulse_sleep,
        # wait (empty if pause_cycles = 0)
        *prog_sleep(pause_cycles),
        # take data
        pins.set_pin(ReadoutPins.ro_frame, True),
        *prog_sleep(frame_cycles),
        pins,
        *pulse_sleep,
    ]

    prog: list[Instruction] = [
        # load count into column register
        pins.set_pin(ReadoutPins.ro_ldcnt, True),
        *pulse_sleep,
        pins,
        *pulse_sleep,
        # shift one bit
        pins.set_pin(ReadoutPins.ro_penable, True),
        *pulse_sleep,
        ShiftOut(1, False),
        *prog_sleep(pulse_cycles + 3, cfg_int),
        # reset counters
        pins.set_pin(ReadoutPins.ro_rescnt, True),
        *pulse_sleep,
        pins,
        *pulse_sleep,
        # start recording frame
        pins_frame,
    ]
    prog_readout = [
        GetTime(),
        # shift out column register contents
        Reset(True, True),
        *prog_shift_dense(setup.encode_column_config(col_cfg), True),
        Sleep(3),
        *pulse_sleep,
        # load config register
        pins_frame.set_pin(ReadoutPins.ro_ldconfig, True),
        *pulse_sleep,
        pins_frame,
    ]
    cycles_readout = count_cycles(prog_readout)
    cycles_remaining = frame_cycles - cycles_readout
    assert cycles_remaining >= 0, 'frame duration too short'
    prog += [
        *prog_readout,
        *prog_sleep(cycles_remaining, cfg_int),
        pins,
        *prog_sleep(pause_cycles, cfg_int),
    ]

    total_cycles = count_cycles(prog)
    frame_us = frame_cycles / frequency
    total_us = total_cycles / frequency
    readout_us = cycles_readout / frequency
    duty_percent = 100*frame_us/total_us
    print(f'frame: {frame_cycles} cycles, {frame_us:.2f} µs ({duty_percent:.1f} %)')
    print(f'readout: {cycles_readout} cycles, {readout_us:.2f} µs')
    print(f'total: {total_cycles} cycles, {total_us:.2f} µs')

    return prog_init, prog


def prog_read_frames_parallel(
    frame_cycles: int,
    pulse_cycles: int,
    pause_cycles: int,
    reset_counters: bool,
    frequency: float,
    setup: HitPixSetup,
    rows: Iterable[int],
    dac_config: HitPix2DacConfig,
) -> tuple[list[Instruction], list[Instruction]]:
    '''returns programs (init and readout)'''
    chip = setup.chip

    cfg_ro = SetCfg(
        shift_rx_invert=True,
        shift_tx_invert=True,
        shift_toggle=True,
        shift_select_dac=False,
        shift_word_len=2 * chip.bits_adder,
        shift_clk_div=setup.clkdiv_ro,
        shift_sample_latency=setup.get_readout_latency(0, frequency),
    )

    cfg_dac = SetCfg(
        shift_rx_invert=False,  # rx not used
        shift_tx_invert=True,
        shift_toggle=False,
        shift_select_dac=True,
        shift_word_len=31,  # rx not used
        shift_clk_div=setup.clkdiv_dac,
        shift_sample_latency=0,  # no data shifted out
    )

    pins = SetPins(0)
    pulse_sleep = prog_sleep(pulse_cycles - 1)

    dac_config_on = dataclasses.replace(dac_config, enable_output_diff=True)
    dac_config_off = dataclasses.replace(dac_config, enable_output_diff=False)

    dac_cfgs = []
    for i in range(setup.chip_columns):
        dac_cfg = bitarray()
        for j in range(setup.chip_columns):
            config = dac_config_on if i == j else dac_config_off
            dac_cfg.extend(config.generate())
        dac_cfgs.append(dac_cfg)

    # init
    col_cfg_init = HitPixColumnConfig(0, 0, 0, -1)
    prog_init: list[Instruction] = [
        Reset(True, True),
        cfg_ro,
        pins,
        *prog_shift_dense(setup.chip.encode_column_config(col_cfg_init), False),
        Sleep(3),
        *pulse_sleep,
        *pins.pulse_pin(ReadoutPins.ro_ldconfig, True, pulse_sleep),
    ]

    # readout
    rows_curr = [-1] + list(rows)
    rows_next = list(rows) + [-1]

    prog: list[Instruction] = [
        cfg_ro
    ]
    # reset counter every frame?
    if reset_counters:
        prog.extend(
            pins.pulse_pin(ReadoutPins.ro_rescnt, True, pulse_sleep)
        )

    prog.extend([
        # wait (empty if pause_cycles = 0)
        *prog_sleep(pause_cycles),
        # take data
        pins.set_pin(ReadoutPins.ro_frame, True),
        *prog_sleep(frame_cycles),
        pins,
        *pulse_sleep,
        # load count into registers
        *pins.pulse_pin(ReadoutPins.ro_ldcnt, True, pulse_sleep),
    ])

    for dac_cfg in dac_cfgs:
        # select chip with DAC
        prog.extend([
            cfg_dac,
            Reset(True, True),
            *pulse_sleep,
            *prog_shift_dense(dac_cfg, False),
            *pulse_sleep,
            *pins.pulse_pin(ReadoutPins.dac_ld, True, pulse_sleep),
            cfg_ro,
        ])
        for row_curr, row_next in zip(rows_curr, rows_next):
            shift_col_cfg = HitPixColumnConfig(0, 0, 0, row_next)
            shift_shiftout = False
            # read out data when the currently selected row is not -1
            if row_curr != -1:
                shift_shiftout = True
                pins_penable = pins.set_pin(ReadoutPins.ro_penable, True)
                prog.extend([
                    # add time to every row to make readout more consistent
                    GetTime(),
                    # shift one bit
                    pins_penable,
                    *pulse_sleep,
                    ShiftOut(1, False),
                    *prog_sleep(pulse_cycles + 3),
                    pins,
                    *pulse_sleep,
                ])
            # shift in + out data
            prog.extend([
                Reset(True, True),
                *prog_shift_dense(setup.chip.encode_column_config(shift_col_cfg), shift_shiftout),
                Sleep(3),
                *pulse_sleep,
                *pins.pulse_pin(ReadoutPins.ro_ldconfig, True, pulse_sleep),
            ])
    # make sure that all data is sent
    # (otherwise readout packets could end up short with small pulse_cycles)
    prog.append(Sleep(8))
    return prog_init, prog


def prog_read_adders_parallel(
    frame_cycles: int,
    pulse_cycles: int,
    pause_cycles: int,
    frequency: float,
    setup: HitPixSetup,
    dac_config: HitPix2DacConfig,
) -> tuple[list[Instruction], list[Instruction]]:
    '''returns programs (init and readout)'''
    chip = setup.chip

    cfg_ro = SetCfg(
        shift_rx_invert=True,
        shift_tx_invert=True,
        shift_toggle=True,
        shift_select_dac=False,
        shift_word_len=2 * chip.bits_adder,
        shift_clk_div=setup.clkdiv_ro,
        shift_sample_latency=setup.get_readout_latency(0, frequency),
    )

    cfg_dac = SetCfg(
        shift_rx_invert=False,  # rx not used
        shift_tx_invert=True,
        shift_toggle=False,
        shift_select_dac=True,
        shift_word_len=31,  # rx not used
        shift_clk_div=setup.clkdiv_dac,
        shift_sample_latency=0,  # no data shifted out
    )

    pins = SetPins(0).set_pin(ReadoutPins.ro_psel, True)
    pins_frame = pins.set_pin(ReadoutPins.ro_frame, True)
    pulse_sleep = prog_sleep(pulse_cycles - 1)

    dac_config_on = dataclasses.replace(dac_config, enable_output_diff=True)
    dac_config_off = dataclasses.replace(dac_config, enable_output_diff=False)

    dac_cfgs = []
    for i in range(setup.chip_columns):
        dac_cfg = bitarray()
        for j in range(setup.chip_columns):
            config = dac_config_on if i == j else dac_config_off
            dac_cfg.extend(config.generate())
        dac_cfgs.append(dac_cfg)

    # init
    col_cfg = HitPixColumnConfig(0, 0, 0, -1)
    prog_init = [
        pins,
        cfg_ro,
        Reset(True, True),
        *prog_shift_dense(setup.chip.encode_column_config(col_cfg), False),
        Sleep(15),
        pins.set_pin(ReadoutPins.ro_ldconfig, True),
        *pulse_sleep,
        pins,
        # reset counters
        pins.set_pin(ReadoutPins.ro_rescnt, True),
        *pulse_sleep,
        pins,
        *pulse_sleep,
        # wait (empty if pause_cycles = 0)
        *prog_sleep(pause_cycles),
        # take data
        pins.set_pin(ReadoutPins.ro_frame, True),
        *prog_sleep(frame_cycles),
        pins,
        *pulse_sleep,
    ]

    prog: list[Instruction] = [
        # load count into column register
        *pins.pulse_pin(ReadoutPins.ro_ldcnt, True, pulse_sleep),
        # shift one bit
        pins.set_pin(ReadoutPins.ro_penable, True),
        *pulse_sleep,
        ShiftOut(1, False),
        *prog_sleep(pulse_cycles + 3, cfg_ro),
        # reset counters
        pins.set_pin(ReadoutPins.ro_rescnt, True),
        *pulse_sleep,
        pins,
        *pulse_sleep,
        # start recording frame
        pins_frame,
    ]
    prog_readout = []
    for dac_cfg in dac_cfgs:
        prog_readout.extend([
            # select chip with DAC
            cfg_dac,
            Reset(True, True),
            *pulse_sleep,
            *prog_shift_dense(dac_cfg, False),
            *pulse_sleep,
            pins.pulse_pin(ReadoutPins.dac_ld, True, pulse_sleep),
            # always add time for consistency
            GetTime(),
            # shift out column register contents
            cfg_ro,
            Reset(True, True),
            *prog_shift_dense(setup.chip.encode_column_config(col_cfg), True),
            Sleep(3),
            *pulse_sleep,
            # load config register
            pins_frame.set_pin(ReadoutPins.ro_ldconfig, True),
            *pulse_sleep,
            pins_frame,
        ])

    cycles_readout = count_cycles(prog_readout)
    cycles_remaining = frame_cycles - cycles_readout
    assert cycles_remaining >= 0, 'frame duration too short'
    prog += [
        *prog_readout,
        *prog_sleep(cycles_remaining, cfg_ro),
        pins,
        *prog_sleep(pause_cycles, cfg_ro),
    ]

    total_cycles = count_cycles(prog)
    frame_us = frame_cycles / frequency
    total_us = total_cycles / frequency
    readout_us = cycles_readout / frequency
    duty_percent = 100*frame_us/total_us
    print(f'frame: {frame_cycles} cycles, {frame_us:.2f} µs ({duty_percent:.1f} %)')
    print(f'readout: {cycles_readout} cycles, {readout_us:.2f} µs')
    print(f'total: {total_cycles} cycles, {total_us:.2f} µs')

    return prog_init, prog
