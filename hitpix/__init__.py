from dataclasses import dataclass
from enum import IntEnum
from typing import Callable
import bitarray
import bitarray.util
from dataclasses import dataclass


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

class HitPixDacConfig:
    def __init__(self, **kwargs) -> None:
        raise NotImplementedError()

    @staticmethod
    def default() -> 'HitPixDacConfig':
        raise NotImplementedError()
    
    def generate(self) -> bitarray.bitarray:
        raise NotImplementedError()


@dataclass
class HitPixVersion:
    rows: int
    columns: int
    bits_counter: int
    encode_column_config: Callable[[HitPixColumnConfig], bitarray.bitarray]
    version_number: int
    dac_config_class: type[HitPixDacConfig]

    @property
    def bits_adder(self) -> int:
        return self.bits_counter + self.rows.bit_length()


@dataclass
class HitPixSetup:
    chip: HitPixVersion
    chip_rows: int  # should always be 1 for now
    chip_columns: int
    invert_pins: int
    version_number: int

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

@dataclass
class HitPix1DacConfig(HitPixDacConfig):
    blres:  int
    vn1:    int
    vnfb:   int
    vnfoll: int
    vndell: int
    vn2:    int
    vnbias: int
    vpload: int
    vncomp: int
    vpfoll: int
    q0:     int = 0x01 # chip doesn't work without these
    qon:    int = 0x05 # just leave them constant

    @staticmethod
    def default() -> 'HitPix1DacConfig':
        import hitpix.defaults
        return HitPix1DacConfig(**hitpix.defaults.dac_default_hitpix1)

    def generate(self) -> bitarray.bitarray:
        assert self.q0 in range(1 << 2)
        assert self.qon in range(1 << 4)
        assert self.blres in range(1 << 6)
        assert self.vn1 in range(1 << 6)
        assert self.vnfb in range(1 << 6)
        assert self.vnfoll in range(1 << 6)
        assert self.vndell in range(1 << 6)
        assert self.vn2 in range(1 << 6)
        assert self.vnbias in range(1 << 6)
        assert self.vpload in range(1 << 6)
        assert self.vncomp in range(1 << 6)
        assert self.vpfoll in range(1 << 6)

        b = bitarray.bitarray()
        b.extend(bitarray.util.int2ba(self.vpfoll, 6, endian='little'))
        b.extend(bitarray.util.int2ba(self.vncomp, 6, endian='little'))
        b.extend(bitarray.util.int2ba(self.vpload, 6, endian='little'))
        b.extend(bitarray.util.int2ba(self.vnbias, 6, endian='little'))
        b.extend(bitarray.util.int2ba(self.vn2, 6, endian='little'))
        b.extend(bitarray.util.int2ba(self.vndell, 6, endian='little'))
        b.extend(bitarray.util.int2ba(self.vnfoll, 6, endian='little'))
        b.extend(bitarray.util.int2ba(self.vnfb, 6, endian='little'))
        b.extend(bitarray.util.int2ba(self.vn1, 6, endian='little'))
        b.extend(bitarray.util.int2ba(self.blres, 6, endian='little'))
        b.extend(bitarray.util.int2ba(self.qon, 4, endian='little'))
        b.extend(bitarray.util.int2ba(self.q0, 2, endian='little'))

        return b

hitpix1 = HitPixVersion(
    rows=24,
    columns=24,
    bits_counter=8,
    encode_column_config=encode_column_config_hitpix1,
    version_number=1,
    dac_config_class=HitPix1DacConfig,
)

hitpix1_single = HitPixSetup(
    chip=hitpix1,
    chip_rows=1,
    chip_columns=1,
    invert_pins=bitfield(ReadoutPins.ro_ldconfig, ReadoutPins.dac_ld,
                         ReadoutPins.dac_inv_ck, ReadoutPins.ro_inv_ck),
    version_number=1,
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

    b = bitarray.bitarray('0') * (14*48)
    b[8:14] = bitarray.util.int2ba(rowaddr, 6, endian='little')
    b[6::14] = bitarray.util.int2ba(conf.ampout_col, 48, endian='little')
    b[7::14] = bitarray.util.int2ba(conf.inject_col, 48, endian='little')

    b_inject_row = bitarray.util.int2ba(conf.inject_row, 48, endian='little')
    for i in range(9):
        i_cfg = 8 + 13 * (i + 1)
        i_inj = i * 6
        inj_sub = b_inject_row[i_inj:i_inj+6]
        b[i_cfg:i_cfg+len(inj_sub)] = inj_sub

    b.reverse()
    return b

@dataclass
class HitPix2DacConfig(HitPixDacConfig):
    # Q[0:1]
    enable_output_cmos : bool
    enable_output_diff : bool
    # Qon[0:3]
    enable_bandgap : bool
    unlock:    int # must be 0x05
    # DAC[0:5]
    iblres:  int
    vn1:    int
    infb:   int
    vnfoll: int
    vndel: int
    vn2:    int
    infb2: int
    ipload2: int
    vncomp: int
    ipfoll: int
    # QVDAC[0:7]
    vth: int

    @staticmethod
    def default() -> 'HitPix2DacConfig':
        import hitpix.defaults
        return HitPix2DacConfig(**hitpix.defaults.dac_default_hitpix2)

    def generate(self) -> bitarray.bitarray:
        assert self.unlock in range(1 << 3)
        assert self.iblres in range(1 << 6)
        assert self.vn1 in range(1 << 6)
        assert self.infb in range(1 << 6)
        assert self.vnfoll in range(1 << 6)
        assert self.vndel in range(1 << 6)
        assert self.vn2 in range(1 << 6)
        assert self.infb2 in range(1 << 6)
        assert self.ipload2 in range(1 << 6)
        assert self.vncomp in range(1 << 6)
        assert self.ipfoll in range(1 << 6)
        assert self.vth in range(1 << 8)

        b = bitarray.bitarray()
        b.extend(bitarray.util.int2ba(self.vth, 8, endian='little'))
        b.extend(bitarray.util.int2ba(self.ipfoll, 6, endian='little'))
        b.extend(bitarray.util.int2ba(self.vncomp, 6, endian='little'))
        b.extend(bitarray.util.int2ba(self.ipload2, 6, endian='little'))
        b.extend(bitarray.util.int2ba(self.infb2, 6, endian='little'))
        b.extend(bitarray.util.int2ba(self.vn2, 6, endian='little'))
        b.extend(bitarray.util.int2ba(self.vndel, 6, endian='little'))
        b.extend(bitarray.util.int2ba(self.vnfoll, 6, endian='little'))
        b.extend(bitarray.util.int2ba(self.infb, 6, endian='little'))
        b.extend(bitarray.util.int2ba(self.vn1, 6, endian='little'))
        b.extend(bitarray.util.int2ba(self.iblres, 6, endian='little'))
        b.extend(bitarray.util.int2ba(self.unlock, 3, endian='little'))
        b.append(self.enable_bandgap)
        b.append(self.enable_output_diff)
        b.append(self.enable_output_cmos)

        return b


hitpix2 = HitPixVersion(
    rows=48,
    columns=48,
    bits_counter=8,
    encode_column_config=encode_column_config_hitpix2,
    version_number=2,
    dac_config_class=HitPix2DacConfig,
)

hitpix2_single = HitPixSetup(
    chip=hitpix2,
    chip_rows=1,
    chip_columns=1,
    invert_pins=bitfield(ReadoutPins.ro_ldconfig, ReadoutPins.dac_ld,
                         ReadoutPins.dac_inv_ck, ReadoutPins.ro_inv_ck),
    version_number=1,
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
    version_number=2,
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
