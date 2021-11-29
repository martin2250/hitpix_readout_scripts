from typing import Union
from statemachine import *
import bitarray, bitarray.util
import numpy as np
import unittest
from dataclasses import dataclass
from hitpix1 import HitPix1Pins, HitPix1ColumnConfig




def prog_shift_simple(data_tx: bitarray.bitarray, shift_out: bool) -> list[Instruction]:
    '''shift in data_tx, data_tx[0] first'''
    prog = []
    data_tx = data_tx.copy()
    while len(data_tx) > 0:
        chunk_size = min(len(data_tx), 16)
        chunk_int = bitarray.util.ba2int(data_tx[:chunk_size])
        chunk_int <<= 16 - chunk_size # left-align
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
            chunk_int <<= 16 - len(data_tx) # left-align
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
        trailing_zeros = data_tx.index(1, 16) if (1 in data_tx[16:]) else len(data_tx)
        # limit shift counter to 8 bits (limit of instruction set)
        trailing_zeros = min(trailing_zeros, 1 << 8)
        if trailing_zeros > 24:
            chunk_int = bitarray.util.ba2int(data_tx[:16])
            del data_tx[:trailing_zeros]
            prog.append(ShiftIn16(trailing_zeros, shift_out, chunk_int))
            continue
        # use ShiftIn24 to send next chunk of bits
        chunk_size = min(len(data_tx), 24)
        assert chunk_size > 16 # should be handled by if above
        chunk_int = bitarray.util.ba2int(data_tx[:chunk_size])
        chunk_int <<= 24 - chunk_size # left-align
        del data_tx[:chunk_size]
        prog.append(ShiftIn24(chunk_size, shift_out, chunk_int))
    return prog

@dataclass
class TestInjection:
    num_injections: int
    shift_clk_div: int
    
    @staticmethod
    def _get_cfg_col_injection(inject_row: int, readout_row: int, odd_pixels: bool) -> HitPix1ColumnConfig:
        mask = int(('01' if odd_pixels else '10') * 12, 2)
        # readout inactive (col 24)
        return HitPix1ColumnConfig(
            inject_row = (1 << inject_row),
            inject_col = mask,
            ampout_col = 0,
            rowaddr = readout_row,
        )

    def prog_test(self) -> list[Instruction]:
        cfg_int = SetCfg(
            shift_rx_invert = True,
            shift_tx_invert = True,
            shift_toggle = True,
            shift_select_dac = False,
            shift_word_len = 2 * 13,
            shift_clk_div = self.shift_clk_div,
            pins = 0,
        )
        prog = []
        cfg_col_prep = self._get_cfg_col_injection(0, 24, False)
        prog.extend([
            cfg_int,
            Sleep(100),
            Reset(True, True),
            *prog_shift_dense(cfg_col_prep.generate(), False),
            Sleep(100),
            cfg_int.set_pin(HitPix1Pins.ro_ldconfig, True),
            Sleep(100),
            cfg_int,
        ])
        for row in range(2*24):
            # read out current row after injections
            cfg_col_readout = HitPix1ColumnConfig(0, 0, 0, row % 24)
            # inject into next row in next loop iteration
            row_inj_next = (row + 1) % 24
            odd_pixels_next = (row + 1) >= 24
            cfg_col_inj_next = self._get_cfg_col_injection(row_inj_next, 24, odd_pixels_next)
            prog.extend([
                # reset counter
                cfg_int.set_pin(HitPix1Pins.ro_rescnt, True),
                Sleep(100),
                cfg_int,
                Sleep(100),
                # set frame high and inject
                cfg_int.set_pin(HitPix1Pins.ro_frame, True),
                Sleep(100),
                Inject(self.num_injections),
                Sleep(100),
                cfg_int,
                Sleep(100),
                # shift in column config for readout
                Reset(True, True),
                *prog_shift_dense(cfg_col_readout.generate(), False),
                Sleep(100),
                cfg_int.set_pin(HitPix1Pins.ro_ldconfig, True),
                Sleep(100),
                cfg_int,
                Sleep(100),
                # load count into column register
                cfg_int.set_pin(HitPix1Pins.ro_ldcnt, True),
                Sleep(100),
                cfg_int,
                Sleep(100),
                cfg_int.set_pin(HitPix1Pins.ro_penable, True),
                Sleep(100),
                ShiftOut(1, False),
                Sleep(100),
                cfg_int,
                Sleep(100),
                # add time to make readout more consistent
                GetTime(),
                # read out data while shifting in configuration for the next round of injections
                Reset(True, True),
                *prog_shift_dense(cfg_col_inj_next.generate(), True),
                Sleep(100),
                cfg_int.set_pin(HitPix1Pins.ro_ldconfig, True),
                Sleep(100),
                cfg_int,
                Sleep(100),
            ])
        prog.append(Finish())
        return prog

def prog_dac_config(cfg_dac_bin: bitarray.bitarray, shift_clk_div: int = 7) -> list[Instruction]:
    cfg_int = SetCfg(
        shift_rx_invert = False, # rx not used
        shift_tx_invert = True,
        shift_toggle = False,
        shift_select_dac = True,
        shift_word_len = 31, # rx not used
        shift_clk_div = shift_clk_div,
        pins = 0,
    )
    return [
        cfg_int,
        Reset(True, True),
        Sleep(100),
        *prog_shift_dense(cfg_dac_bin, False),
        Sleep(100),
        cfg_int.set_pin(HitPix1Pins.dac_ld, True),
        Sleep(100),
        cfg_int,
    ]

def prog_rescnt() -> list[Instruction]:
    cfg_int = SetCfg(
        shift_rx_invert = True,
        shift_tx_invert = True,
        shift_toggle = True,
        shift_select_dac = False,
        shift_word_len = 2 * 13,
        shift_clk_div = 0,
        pins = 0,
    )
    return [
        cfg_int.set_pin(HitPix1Pins.ro_rescnt, True),
        Sleep(10),
        cfg_int,
    ]

def prog_inject(num_injections: int, cfg_col: Union[HitPix1ColumnConfig, bitarray.bitarray]) -> list[Instruction]:
    cfg_int = SetCfg(
        shift_rx_invert = True,
        shift_tx_invert = True,
        shift_toggle = True,
        shift_select_dac = False,
        shift_word_len = 2 * 13,
        shift_clk_div = 2,
        pins = 0,
    )
    cfg_col_bit = cfg_col if isinstance(cfg_col, bitarray.bitarray) else cfg_col.generate()
    return [
        cfg_int.set_pin(HitPix1Pins.ro_rescnt, True),
        Sleep(10),
        cfg_int,
        Sleep(10),
        Reset(True, True),
        *prog_shift_dense(cfg_col_bit, False),
        Sleep(200),
        cfg_int.set_pin(HitPix1Pins.ro_ldconfig, True),
        Sleep(200),
        cfg_int,
        Sleep(500),
        cfg_int.set_pin(HitPix1Pins.ro_frame, True),
        Sleep(300),
        Inject(num_injections),
        Sleep(300),
        cfg_int,
        Sleep(300),
    ]

def prog_full_readout(shift_clk_div: int) -> list[Instruction]:
    cfg_int = SetCfg(
        shift_rx_invert = True,
        shift_tx_invert = True,
        shift_toggle = True,
        shift_select_dac = False,
        shift_word_len = 2 * 13,
        shift_clk_div = shift_clk_div,
        pins = 0,
    )

    prog: list[Instruction] = [
        cfg_int,
    ]

    for row in range(25):
        col_cfg = HitPix1ColumnConfig(0, 0, 0, row)
        # add time to make readout more consistent
        if row > 0:
            prog.append(GetTime())
        prog.extend([
            Sleep(100),
            Reset(True, True),
            *prog_shift_dense(col_cfg.generate(), row > 0),
            Sleep(100),
            cfg_int.set_pin(HitPix1Pins.ro_ldconfig, True),
            Sleep(100),
            cfg_int,
            Sleep(100),
        ])
        if row == 24:
            break
        prog.extend([
            # load count into column register
            cfg_int.set_pin(HitPix1Pins.ro_ldcnt, True),
            Sleep(100),
            cfg_int,
            Sleep(100),
            cfg_int.set_pin(HitPix1Pins.ro_penable, True),
            Sleep(100),
            ShiftOut(1, False),
            Sleep(100),
            cfg_int,
            Sleep(100),
        ])
    
    return prog

def decode_column_packets(packet: bytes, columns: int = 24, bits_shift: int = 13, bits_mask: int = 8) -> tuple[np.ndarray, np.ndarray]:
    '''decode column packets with timestamps'''
    # check that packet has right size for 32 bit ints
    assert (len(packet) % 4) == 0
    data = np.frombuffer(packet, '<u4')
    # check that number of values can be divided into equal parts
    assert (len(data) % (columns // 2 + 1)) == 0
    # reshape data to extract individual hit frames
    data = np.reshape(data, (-1, columns // 2 + 1))
    # get timestamps
    timestamps = data[:, 0]
    # get hits
    hits_raw = data[:, 1:].flat
    # duplicate all numbers to extract both <bits> bit values
    bit_mask = (1 << bits_mask) - 1
    hits = np.dstack((
        np.bitwise_and(np.right_shift(hits_raw, bits_shift), bit_mask),
        np.bitwise_and(hits_raw, bit_mask),
    ))
    hits = np.reshape(hits, (-1, columns))
    return timestamps, hits

class DecodeTest(unittest.TestCase):
    def test_decode(self):
        packet = b''.join(n.to_bytes(4, 'little') for n in [
            82374, # timestamp
            (14 << 13) | 77, # pixel 0 & 1
            (36 << 13) | 221, # pixel 2 & 3
            568734, # timestamp
            (25 << 13) | 96, # pixel 0 & 1
            (66 << 13) | 47, # pixel 2 & 3
        ])
        timestamps, hits = decode_column_packets(packet, 4)
        self.assertEqual(timestamps.shape, (2,))
        self.assertEqual(hits.shape, (2, 4))
        self.assertEqual(list(timestamps), [82374, 568734])
        self.assertEqual(list(hits[0]), [14, 77, 36, 221])
        self.assertEqual(list(hits[1]), [25, 96, 66, 47])

if __name__ == '__main__':
    # unittest.main()
    prog = TestInjection(64, 2).prog_test()
    for instr in prog:
        print(instr)
    print(len(prog))