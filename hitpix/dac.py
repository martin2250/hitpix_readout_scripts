from dataclasses import dataclass
import bitarray, bitarray.util

@dataclass
class HitPixDacConfig:
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
    def default() -> 'HitPixDacConfig':
        import hitpix.defaults
        return HitPixDacConfig(**hitpix.defaults.dac_default_hitpix1)

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
