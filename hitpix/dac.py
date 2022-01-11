from dataclasses import dataclass
import bitarray, bitarray.util

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

@dataclass
class HitPix2DacConfig:
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
