from dataclasses import dataclass
import bitarray, bitarray.util

def __fill__(name: str, bits: int, names: list[str], default):
    one = bitarray.bitarray('1')
    for i in range(bits):
        c = default()
        c.__setattr__(name, 1 << i)
        b = c.generate()
        if one not in b:
            print(f'error filling bit {i} of {name}')
        names[b.index(one)] = f'{name}[{i}]'

@dataclass
class ColumnConfig:
    inject_row: int
    inject_col: int
    ampout_col: int
    rowaddr: int

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

    @staticmethod
    def test() -> list[str]:
        names = ['' for _ in range(13*24)]
        default = lambda: ColumnConfig(0, 0, 0, 0)
        __fill__('inject_row', 24, names, default)
        __fill__('inject_col', 24, names, default)
        __fill__('ampout_col', 24, names, default)
        __fill__('rowaddr', 5, names,  default)
        return names


@dataclass
class DacConfig:
    q0: int
    qon: int
    blres: int
    vn1: int
    vnfb: int
    vnfoll: int
    vndell: int
    vn2: int
    vnbias: int
    vpload: int
    vncomp: int
    vpfoll: int

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
        b.extend(bitarray.util.int2ba(self.vpfoll, 6, endian='big'))
        b.extend(bitarray.util.int2ba(self.vncomp, 6, endian='big'))
        b.extend(bitarray.util.int2ba(self.vpload, 6, endian='big'))
        b.extend(bitarray.util.int2ba(self.vnbias, 6, endian='big'))
        b.extend(bitarray.util.int2ba(self.vn2, 6, endian='big'))
        b.extend(bitarray.util.int2ba(self.vndell, 6, endian='big'))
        b.extend(bitarray.util.int2ba(self.vnfoll, 6, endian='big'))
        b.extend(bitarray.util.int2ba(self.vnfb, 6, endian='big'))
        b.extend(bitarray.util.int2ba(self.vn1, 6, endian='big'))
        b.extend(bitarray.util.int2ba(self.blres, 6, endian='big'))
        b.extend(bitarray.util.int2ba(self.qon, 4, endian='little'))
        b.extend(bitarray.util.int2ba(self.q0, 2, endian='little'))

        return b
    
    @staticmethod
    def test() -> list[str]:
        names = ['' for _ in range(66)]
        default = lambda: DacConfig(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        __fill__('q0', 2, names, default)
        __fill__('qon', 4, names, default)
        __fill__('blres', 6, names, default)
        __fill__('vn1', 6, names,  default)
        __fill__('vnfb', 6, names, default)
        __fill__('vnfoll', 6, names, default)
        __fill__('vndell', 6, names, default)
        __fill__('vn2', 6, names,  default)
        __fill__('vnbias', 6, names, default)
        __fill__('vpload', 6, names, default)
        __fill__('vncomp', 6, names, default)
        __fill__('vpfoll', 6, names,  default)
        return names


if __name__ == '__main__':
    names = DacConfig.test()
    for i, name in enumerate(names):
        print(f'{i:3d} {name}')