from typing import Iterable
import hitpix1_config
from statemachine import *
import bitarray, bitarray.util
import dataclasses
import numpy as np
import unittest

def prog_shift_dense(data_tx: bitarray.bitarray, shift_out: bool) -> list[Instruction]:
    '''shift in data_tx, data_tx[0] first'''
    prog = []
    chunk_size = 24
    data_tx = data_tx.copy()
    while len(data_tx) > 0:
        # fewer than or exactly 16 bits remaining? use ShiftIn16
        if len(data_tx) <= 16:
            chunk_int = bitarray.util.ba2int(data_tx[:chunk_size])
            chunk_int <<= 16 - len(data_tx) # left-align
            prog.append(ShiftIn16(chunk_size, shift_out, chunk_int))
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
class InjectionTest:
    num_injections: int
    shift_clk_div: int
    
    @staticmethod
    def _get_cfg_col_injection(inject_row: int, readout_row: int, odd_pixels: bool) -> hitpix1_config.ColumnConfig:
        mask = int(('01' if odd_pixels else '10') * 12, 2)
        # readout inactive (col 24)
        return hitpix1_config.ColumnConfig(
            inject_row = (1 << inject_row),
            inject_col = mask,
            ampout_col = 0,
            rowaddr = readout_row,
        )

    def prog_test(self) -> list[Instruction]:
        cfg_int = SetCfg(
            shift_rx_invert = False,
            shift_tx_invert = False,
            shift_toggle = True,
            shift_select_dac = False,
            shift_word_len = 2 * 13,
            shift_clk_div = self.shift_clk_div,
            pins = 0,
        )
        prog = []
        cfg_col_prep = self._get_cfg_col_injection(0, 24, False)
        prog.extend([
            Reset(True, True),
            *prog_shift_dense(cfg_col_prep.generate(), False),
            cfg_int.set_pin(HitPix1Pins.ro_ldconfig, True),
            cfg_int,
        ])
        for row in range(2*24):
            # read out current row after injections
            cfg_col_readout = hitpix1_config.ColumnConfig(0, 0, 0, row % 24)
            # inject into next row in next loop iteration
            row_inj_next = (row + 1) % 24
            odd_pixels_next = (row + 1) >= 24
            cfg_col_inj_next = self._get_cfg_col_injection(row_inj_next, 24, odd_pixels_next)
            prog.extend([
                # reset counter
                cfg_int.set_pin(HitPix1Pins.ro_rescnt, True),
                cfg_int,
                # set frame high and inject
                cfg_int.set_pin(HitPix1Pins.ro_frame, True),
                Inject(self.num_injections),
                cfg_int,
                # shift in column config for readout
                Reset(True, True),
                *prog_shift_dense(cfg_col_readout.generate(), False),
                cfg_int.set_pin(HitPix1Pins.ro_ldconfig, True),
                cfg_int,
                # load count into column register
                cfg_int.set_pin(HitPix1Pins.ro_ldcnt, True),
                cfg_int,
                cfg_int.set_pin(HitPix1Pins.ro_penable, True),
                ShiftOut(1, False),
                cfg_int,
                # add time to make readout more consistent
                GetTime(),
                # read out data while shifting in configuration for the next round of injections
                Reset(True, True),
                *prog_shift_dense(cfg_col_inj_next.generate(), True),
                cfg_int.set_pin(HitPix1Pins.ro_ldconfig, True),
                cfg_int,
            ])
        return prog

def prog_dac_config(cfg_dac: hitpix1_config.DacConfig, cfg: SetCfg) -> list[Instruction]:
    cfg_int = cfg.modify(
        shift_tx_invert = False,
        shift_toggle = False,
        shift_select_dac = True,
        shift_clk_div = 5,
    ).set_pin(HitPix1Pins.dac_ld, False)
    prog = []
    prog.append(cfg_int)
    prog.append(Sleep(100))
    prog.extend(prog_shift_dense(cfg_dac.generate(), False))
    prog.append(Sleep(100))
    prog.append(cfg_int.set_pin(HitPix1Pins.dac_ld, True))
    prog.append(Sleep(100))
    prog.append(cfg)
    return prog

def decode_column_packets(packet: bytes, columns: int = 24, bits: int = 13) -> tuple[np.ndarray, np.ndarray]:
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
    bit_mask = (1 << bits) - 1
    hits = np.dstack((
        np.bitwise_and(np.right_shift(hits_raw, 13), bit_mask),
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
    prog = InjectionTest(64, 2).prog_test()
    for instr in prog:
        print(instr)
    print(len(prog))