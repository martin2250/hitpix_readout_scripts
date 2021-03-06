from typing import Optional
import bitarray
import bitarray.util
import numpy as np

from .instructions import *
from hitpix import HitPixColumnConfig, HitPixSetup, ReadoutPins


def prog_shift_simple(data_tx: bitarray.bitarray, shift_out: bool) -> list[Instruction]:
    '''shift in data_tx, data_tx[0] first'''
    prog = []
    data_tx = data_tx.copy()
    while len(data_tx) > 0:
        chunk_size = min(len(data_tx), 16)
        chunk_int = bitarray.util.ba2int(data_tx[:chunk_size])
        chunk_int <<= 16 - chunk_size  # left-align
        del data_tx[:chunk_size]
        prog.append(ShiftIn16(chunk_size, shift_out, chunk_int))
    return prog


def prog_shift_dense(data_tx: bitarray.bitarray, shift_out: bool) -> list[Instruction]:
    '''shift in data_tx, data_tx[0] first'''
    prog = []
    data_tx = data_tx.copy()
    while len(data_tx) > 0:
        # fewer than or exactly 16 bits remaining? use ShiftIn16
        if len(data_tx) <= 16:
            chunk_int = bitarray.util.ba2int(data_tx)
            chunk_int <<= 16 - len(data_tx)  # left-align
            prog.append(ShiftIn16(len(data_tx), shift_out, chunk_int))
            break
        # check for long run of zeros at the start of the array
        leading_zeros = data_tx.index(1) if (1 in data_tx) else len(data_tx)
        # more zeros than fit into one data word? use ShiftOut
        if leading_zeros > 24:
            prog.append(ShiftOut(leading_zeros, shift_out))
            del data_tx[:leading_zeros]
            continue
        # only few ones and then a long run of zeros?
        # use ShiftIn16 with more num_bits than data
        trailing_zeros = data_tx.index(1, 16) if (
            1 in data_tx[16:]) else len(data_tx)
        # limit shift counter to 8 bits (limit of instruction set)
        trailing_zeros = min(trailing_zeros, 1 << 8)
        if trailing_zeros > 24:
            chunk_int = bitarray.util.ba2int(data_tx[:16])
            del data_tx[:trailing_zeros]
            prog.append(ShiftIn16(trailing_zeros, shift_out, chunk_int))
            continue
        # use ShiftIn24 to send next chunk of bits
        chunk_size = min(len(data_tx), 24)
        assert chunk_size > 16  # should be handled by if above
        chunk_int = bitarray.util.ba2int(data_tx[:chunk_size])
        chunk_int <<= 24 - chunk_size  # left-align
        del data_tx[:chunk_size]
        prog.append(ShiftIn24(chunk_size, shift_out, chunk_int))
    return prog


def prog_sleep(sleep_cycles: int, single_cycle: Optional[Instruction] = None) -> list[Instruction]:
    # handle simple cases
    if sleep_cycles == 0:
        return []
    if sleep_cycles == 1:
        assert single_cycle is not None
        return [single_cycle]
    # handle other cases
    cycles_max = (1 << 24) + 1
    prog = []
    while sleep_cycles > 0:
        cycles = min(sleep_cycles, cycles_max)
        sleep_cycles -= cycles
        # 1 cycle remaining? not supported without single cycle
        # -> borrow one cycle for the next sleep instruction
        if sleep_cycles == 1:
            cycles -= 1
            sleep_cycles += 1
        prog.append(Sleep(cycles))
    return prog


def decode_column_packets(packet: bytes, columns: int = 24, bits_shift: int = 13, bits_mask: Optional[int] = 8) -> tuple[np.ndarray, np.ndarray]:
    '''decode column packets with timestamps'''
    # check that packet has right size for 32 bit ints
    assert (len(packet) % 4) == 0
    data = np.frombuffer(packet, '<u4')
    # check that number of values can be divided into equal parts
    assert (len(data) % (columns // 2 + 1)) == 0, f'{len(data)=}'
    # reshape data to extract individual hit frames
    data = np.reshape(data, (-1, columns // 2 + 1))
    # get timestamps
    timestamps = data[:, 0]
    # get hits
    hits_raw = data[:, 1:].flat
    # duplicate all numbers to extract both <bits> bit values
    hits = np.dstack((
        np.right_shift(hits_raw, bits_shift),
        hits_raw,
    ))
    if bits_mask is not None:
        bit_mask = (1 << bits_mask) - 1
        hits = np.bitwise_and(hits, bit_mask)
    hits = np.reshape(hits, (-1, columns))
    # counter count down
    return timestamps, hits

def prog_dac_config(cfg_dac_bin: bitarray.bitarray) -> list[Instruction]:
    cfg_int = SetCfg(
        shift_rx_invert = False, # rx not used
        shift_tx_invert = True,
        shift_toggle = False,
        shift_select_dac = True,
        shift_word_len = 31, # rx not used
        shift_clk_div = 2,
        shift_sample_latency=0, # no data shifted out
    )
    pins = SetPins(0)
    return [
        cfg_int,
        pins,
        Reset(True, True),
        Sleep(100),
        *prog_shift_dense(cfg_dac_bin, False),
        Sleep(100),
        pins.set_pin(ReadoutPins.dac_ld, True),
        Sleep(100),
        pins,
    ]

def prog_col_config(cfg_col_bin: bitarray.bitarray, shift_clk_div: int = 0, shift_out_bits: Optional[int] = None) -> list[Instruction]:
    cfg_int = SetCfg(
        shift_rx_invert = True,
        shift_tx_invert = True,
        shift_toggle = True,
        shift_select_dac = False,
        shift_word_len = shift_out_bits or 32,
        shift_clk_div = shift_clk_div,
        shift_sample_latency=0,
    )
    pins = SetPins(0)
    return [
        cfg_int,
        pins,
        Reset(True, True),
        Sleep(50),
        *prog_shift_dense(cfg_col_bin, shift_out_bits is not None),
        Sleep(50),
        pins.set_pin(ReadoutPins.ro_ldconfig, True),
        Sleep(50),
        pins,
    ]

def prog_read_matrix(setup: HitPixSetup, shift_clk_div: int = 1, pulse_cycles: int = 50, rows: Optional[list[int]] = None) -> list[Instruction]:
    assert setup.chip_rows == 1
    chip = setup.chip

    if rows is None:
        rows = list(range(chip.rows))

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
    prog = [
        *prog_col_config(
            cfg_col_bin = setup.encode_column_config(HitPixColumnConfig(0, 0, 0, rows[0])),
            shift_clk_div=shift_clk_div,
            shift_out_bits=None,
        ),
    ]
    rows_next = rows[1:] + [-1]
    for row_next in rows_next:
        # readout
        prog.extend([
            cfg_int,
            pins,
            # prepend row with time
            GetTime(),
            # load count into column register
            pins.set_pin(ReadoutPins.ro_ldcnt, True),
            Sleep(pulse_cycles),
            pins,
            Sleep(pulse_cycles),
            pins.set_pin(ReadoutPins.ro_penable, True),
            Sleep(pulse_cycles),
            ShiftOut(1, False),
            Sleep(pulse_cycles + 3),
            pins,
            Sleep(pulse_cycles),
            # shift out data of current row and shift in next row
            *prog_col_config(
                cfg_col_bin = setup.encode_column_config(HitPixColumnConfig(0, 0, 0, row_next)),
                shift_clk_div=shift_clk_div,
                shift_out_bits=2*chip.bits_adder,
            ),
        ])
    return prog

if __name__ == '__main__':
    import unittest

    class TestProgSleep(unittest.TestCase):
        def test_sleeps(self):
            for xx in range(5):
                for count in range(20):
                    cycles = max(0, xx * (1 << 24) - 10 + count)
                    prog = prog_sleep(cycles, SetPins(0))
                    calc = count_cycles(prog)
                    self.assertEqual(cycles, calc)
                    for instr in prog:
                        instr.to_binary()
    
    unittest.main()