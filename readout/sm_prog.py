import bitarray
import bitarray.util
import numpy as np

from .instructions import *
from hitpix.hitpix1 import HitPix1Pins


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


def prog_sleep(sleep_cycles: int) -> list[Instruction]:
    prog = []
    while sleep_cycles > 0:
        sleep_cycles_i = min(sleep_cycles, 1 << 24)
        prog.append(Sleep(sleep_cycles_i))
        sleep_cycles -= sleep_cycles_i
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
    # counter count down
    return timestamps, hits

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
