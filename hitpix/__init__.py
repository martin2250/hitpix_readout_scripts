from dataclasses import dataclass
import dataclasses
from enum import IntEnum
from multiprocessing.sharedctypes import Value
from typing import Callable

import bitarray
import bitarray.util
import numpy as np


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
    rowaddr: int = -1  # -1 == select first non-existent row


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

    get_readout_latency: Callable[[int, float], int]
    readout_div1_clk1: int
    readout_div1_clk2: int
    readout_div2_clk1: int
    readout_div2_clk2: int

    vc_baseline: tuple[int, int]  # card slot, channel
    vc_threshold: tuple[int, int]  # card slot, channel
    vc_injection: tuple[int, int]  # card slot, channel

    invert_rx: bool = True

    @property
    def pixel_rows(self) -> int:
        return self.chip_rows * self.chip.rows

    @property
    def pixel_columns(self) -> int:
        return self.chip_columns * self.chip.columns

    def encode_column_config(self, conf: HitPixColumnConfig) -> bitarray.bitarray:
        '''simply repeats column config x times'''
        assert self.chip_rows == 1
        data = bitarray.bitarray()
        for col in reversed(range(self.chip_columns)):
            shift = col * self.chip.columns
            mask = (1 << self.chip.columns) - 1
            conf_chip = HitPixColumnConfig(
                inject_row=conf.inject_row,
                inject_col=(conf.inject_col >> shift) & mask,
                ampout_col=(conf.ampout_col >> shift) & mask,
                rowaddr=conf.rowaddr,
            )
            data.extend(self.chip.encode_column_config(conf_chip))
        return data

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


def get_readout_latency_hitpix1(clk_div: int, frequency_mhz: float, use_dac: bool = False) -> int:
    if use_dac:
        raise ValueError('no data for DAC')

    polyvals_ro = {
        0: [3.524e-06, -1.151e-03, 2.005e-01, 1.010e+01],
        1: [1.393e-07, -2.918e-05, 8.649e-02, 1.095e+01],
        2: [3.697e-06, -1.080e-03, 1.388e-01, 6.684e+00],
    }

    if not clk_div in polyvals_ro:
        raise ValueError(f'no latency data available for {clk_div=}')

    polyvals = polyvals_ro[clk_div]
    latency_float = np.poly1d(polyvals)(frequency_mhz)

    return round(latency_float)


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
    q0:     int = 0x01  # chip doesn't work without these
    qon:    int = 0x05  # just leave them constant

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
    get_readout_latency=get_readout_latency_hitpix1,
    # --------------- 0b111111000000,
    readout_div1_clk1=0b001110000000,
    readout_div1_clk2=0b000001110000,
    # --------------- 0b111111111111000000000000,
    readout_div2_clk1=0b000001111100000000000000,
    readout_div2_clk2=0b000000000000011111000000,
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
        i_cfg = 8 + 14 * (i + 1)
        i_inj = i * 6
        inj_sub = b_inject_row[i_inj:i_inj+6]
        b[i_cfg:i_cfg+len(inj_sub)] = inj_sub

    b.reverse()
    return b


def get_readout_latency_hitpix2(clk_div: int, frequency_mhz: float, use_dac: bool = False) -> int:
    if use_dac:
        raise ValueError('no data for DAC')

    polyvals_ro = {
        0: [-6.656e-05, 6.263e-03, -8.489e-03, 1.427e+01],
        1: [4.898e-06, -1.175e-03, 2.210e-01, 9.801e+00],
        2: [-7.081e-06, 1.860e-03, -3.695e-02, 1.070e+01],
    }

    if not clk_div in polyvals_ro:
        raise ValueError(f'no latency data available for {clk_div=}')

    polyvals = polyvals_ro[clk_div]
    latency_float = np.poly1d(polyvals)(frequency_mhz)

    return round(latency_float)


@dataclass
class HitPix2DacConfig(HitPixDacConfig):
    # Q[0:1]
    enable_output_cmos: bool
    enable_output_diff: bool
    # Qon[0:3]
    enable_bandgap: bool
    unlock:    int  # must be 0x05
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
        b.extend(bitarray.util.int2ba(self.vth, 8, endian='big'))
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
    invert_pins=bitfield(
        ReadoutPins.ro_ldconfig,
        ReadoutPins.dac_ld,
        ReadoutPins.dac_inv_ck,
        ReadoutPins.ro_inv_ck,
        ReadoutPins.ro_ldcnt,
    ),
    version_number=1,
    vc_baseline=(0, 4),
    vc_threshold=(0, 1),
    vc_injection=(2, 0),
    get_readout_latency=get_readout_latency_hitpix2,
    # --------------- 0b111111000000,
    readout_div1_clk1=0b001100000000,
    readout_div1_clk2=0b000001100000,
    # --------------- 0b111111111111000000000000,
    readout_div2_clk1=0b000001111100000000000000,
    readout_div2_clk2=0b000000000000011111000000,
)

hitpix2_1x2_last = HitPixSetup(
    chip=hitpix2,
    chip_rows=1,
    chip_columns=2,
    invert_pins=bitfield(
        ReadoutPins.ro_ldconfig,
        ReadoutPins.dac_ld,
        ReadoutPins.dac_inv_ck,
        ReadoutPins.ro_inv_ck,
        ReadoutPins.ro_ldcnt,
    ),
    version_number=2,
    vc_baseline=(0, 0),  # set by voltage divider, not connected!
    vc_threshold=(0, 1),  # not connected!
    vc_injection=(2, 0),
    get_readout_latency=get_readout_latency_hitpix2,
    # --------------- 0b111111000000,
    readout_div1_clk1=0b001100000000,
    readout_div1_clk2=0b000001100000,
    # --------------- 0b111111111111000000000000,
    readout_div2_clk1=0b000001111100000000000000,
    readout_div2_clk2=0b000000000000011111000000,
)

hitpix2_1x2_first = dataclasses.replace(hitpix2_1x2_last, invert_rx=False)

# keep in sync with defaults.py!!
setups = {
    'hitpix1': hitpix1_single,
    'hitpix2-1x1': hitpix2_single,
    'hitpix2-1x2-first': hitpix2_1x2_first,
    'hitpix2-1x2-last': hitpix2_1x2_last,
}
