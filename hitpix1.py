from enum import IntEnum
from dataclasses import dataclass
from readout.readout import Readout
from readout.dac_card import DacCard
import bitarray, bitarray.util

class HitPix1VoltageCards(IntEnum):
    threshold = 1
    baseline  = 4

class HitPix1Pins(IntEnum):
    ro_ldconfig = 0
    ro_psel     = 1
    ro_penable  = 2
    ro_ldcnt    = 3
    ro_rescnt   = 4
    ro_frame    = 5
    dac_ld      = 6
    dac_inv_ck  = 30
    ro_inv_ck   = 31

class HitPix1Readout(Readout):
    def __init__(self, serial_name: str, timeout: float = 0.5) -> None:
        super().__init__(serial_name, timeout=timeout)

        self.voltage_card = DacCard(0, 8, 3.3, 1.8, self)
        self.injection_card = DacCard(2, 2, 3.3, 1.8, self)

        invert_pins = 0
        invert_pins |= 1 << HitPix1Pins.ro_ldconfig
        invert_pins |= 1 << HitPix1Pins.dac_ld
        invert_pins |= 1 << 30 # ck 1&2 dac
        invert_pins |= 1 << 31 # ck 1&2 readout
        self.write_register(self.ADDR_SM_INVERT_PINS, invert_pins)

        self.frequency_mhz = 200
    
    def set_treshold_voltage(self, voltage: float) -> None:
        self.voltage_card.set_voltage(HitPix1VoltageCards.threshold, voltage)
    
    def set_baseline_voltage(self, voltage: float) -> None:
        self.voltage_card.set_voltage(HitPix1VoltageCards.baseline, voltage)

    def set_injection_voltage(self, voltage: float) -> None:
        self.injection_card.set_voltage(0, voltage)

@dataclass
class HitPix1ColumnConfig:
    inject_row: int = 0
    inject_col: int = 0
    ampout_col: int = 0
    rowaddr: int    = 24

    def generate(self) -> bitarray.bitarray:
        assert self.rowaddr in range(1 << 5)
        assert self.inject_col in range(1 << 24)
        assert self.inject_row in range(1 << 24)
        assert self.ampout_col in range(1 << 24)

        b = bitarray.bitarray('0') * (13*24)
        b[8:13] = bitarray.util.int2ba(self.rowaddr, 5, endian='little')
        b[6::13] = bitarray.util.int2ba(self.ampout_col, 24, endian='little')
        b[7::13] = bitarray.util.int2ba(self.inject_col, 24, endian='little')

        b_inject_row = bitarray.util.int2ba(self.inject_row, 24, endian='little')
        for i in range(5):
            i_cfg = 21 + 13 * i
            i_inj = i * 5
            inj_sub = b_inject_row[i_inj:i_inj+5]
            b[i_cfg:i_cfg+len(inj_sub)] = inj_sub

        b.reverse()
        return b


@dataclass
class HitPix1DacConfig:
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
