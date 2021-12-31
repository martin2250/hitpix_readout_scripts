from dataclasses import dataclass
from enum import IntEnum
from typing import Callable
import bitarray
import bitarray.util


class ReadoutPins(IntEnum):
    ro_ldconfig = 0
    ro_psel = 1
    ro_penable = 2
    ro_ldcnt = 3
    ro_rescnt = 4
    ro_frame = 5
    dac_ld = 6
    dac_inv_ck = 30
    ro_inv_ck = 31


@dataclass
class HitPixColumnConfig:
    inject_row: int = 0
    inject_col: int = 0
    ampout_col: int = 0
    rowaddr: int = -1 # -1 == select first non-existent row


@dataclass
class HitPixVersion:
    rows: int
    columns: int
    bits_counter: int
    encode_column_config: Callable[[HitPixColumnConfig], bitarray.bitarray]

    @property
    def bits_adder(self) -> int:
        return self.bits_counter + self.rows.bit_length()


@dataclass
class HitPixSetup:
    chip: HitPixVersion
    chip_rows: int  # should always be 1 for now
    chip_columns: int
    invert_pins: int

    vc_baseline: tuple[int, int]  # card slot, channel
    vc_threshold: tuple[int, int]  # card slot, channel
    vc_injection: tuple[int, int]  # card slot, channel

    @property
    def pixel_rows(self) -> int:
        return self.chip_rows * self.chip.rows
    @property
    def pixel_columns(self) -> int:
        return self.chip_columns * self.chip.columns
    
    def encode_column_config(self, conf: HitPixColumnConfig) -> bitarray.bitarray:
        '''simply repeats column config x times'''
        assert self.chip_rows == 1
        return self.chip.encode_column_config(conf) * self.chip_columns


################################################################################
# helper methods

def bitfield(*indices: int) -> int:
    i = 0
    for idx in indices:
        i |= 1 << idx
    return i

################################################################################
# HitPix1


def encode_column_config_hitpix1(conf: HitPixColumnConfig) -> bitarray.bitarray:
    rowaddr = 24 if conf.rowaddr == -1 else conf.rowaddr
    assert rowaddr in range(1 << 5)
    assert conf.inject_col in range(1 << 24)
    assert conf.inject_row in range(1 << 24)
    assert conf.ampout_col in range(1 << 24)

    b = bitarray.bitarray('0') * (13*24)
    b[8:13] = bitarray.util.int2ba(rowaddr, 5, endian='little')
    b[6::13] = bitarray.util.int2ba(conf.ampout_col, 24, endian='little')
    b[7::13] = bitarray.util.int2ba(conf.inject_col, 24, endian='little')

    b_inject_row = bitarray.util.int2ba(conf.inject_row, 24, endian='little')
    for i in range(5):
        i_cfg = 21 + 13 * i
        i_inj = i * 5
        inj_sub = b_inject_row[i_inj:i_inj+5]
        b[i_cfg:i_cfg+len(inj_sub)] = inj_sub

    b.reverse()
    return b


hitpix1 = HitPixVersion(
    rows=24,
    columns=24,
    bits_counter=8,
    encode_column_config=encode_column_config_hitpix1,
)

hitpix1_single = HitPixSetup(
    chip=hitpix1,
    chip_rows=1,
    chip_columns=1,
    invert_pins=bitfield(ReadoutPins.ro_ldconfig, ReadoutPins.dac_ld,
                         ReadoutPins.dac_inv_ck, ReadoutPins.ro_inv_ck),
    vc_baseline=(0, 4),
    vc_threshold=(0, 1),
    vc_injection=(2, 0),
)

################################################################################
# HitPix2


def encode_column_config_hitpix2(conf: HitPixColumnConfig) -> bitarray.bitarray:
    rowaddr = 48 if conf.rowaddr == -1 else conf.rowaddr
    assert rowaddr in range(1 << 6)
    assert conf.inject_col in range(1 << 48)
    assert conf.inject_row in range(1 << 48)
    assert conf.ampout_col in range(1 << 48)

    raise NotImplementedError()
    b = bitarray.bitarray('0') * (14*48)
    b[8:14] = bitarray.util.int2ba(rowaddr, 6, endian='little')
    b[6::14] = bitarray.util.int2ba(conf.ampout_col, 48, endian='little')
    b[7::14] = bitarray.util.int2ba(conf.inject_col, 48, endian='little')

    b_inject_row = bitarray.util.int2ba(conf.inject_row, 48, endian='little')
    for i in range(5):
        i_cfg = 21 + 13 * i
        i_inj = i * 5
        inj_sub = b_inject_row[i_inj:i_inj+5]
        b[i_cfg:i_cfg+len(inj_sub)] = inj_sub

    b.reverse()
    return b


hitpix2 = HitPixVersion(
    rows=48,
    columns=48,
    bits_counter=8,
    encode_column_config=encode_column_config_hitpix2,
)

hitpix2_single = HitPixSetup(
    chip=hitpix2,
    chip_rows=1,
    chip_columns=1,
    invert_pins=bitfield(ReadoutPins.ro_ldconfig, ReadoutPins.dac_ld,
                         ReadoutPins.dac_inv_ck, ReadoutPins.ro_inv_ck),
    vc_baseline=(0, 4),
    vc_threshold=(0, 1),
    vc_injection=(2, 0),
)

hitpix2_row = HitPixSetup(
    chip=hitpix2,
    chip_rows=1,
    chip_columns=5,
    invert_pins=bitfield(ReadoutPins.ro_ldconfig, ReadoutPins.dac_ld,
                         ReadoutPins.dac_inv_ck, ReadoutPins.ro_inv_ck),
    vc_baseline=(-1, -1),
    vc_threshold=(-1, -1),
    vc_injection=(-1, -1),
)

# keep in sync with defaults.py!!
setups = {
   'hitpix1': hitpix1_single,
   'hitpix2-1x1': hitpix2_single,
   'hitpix2-1x5': hitpix2_row,
}