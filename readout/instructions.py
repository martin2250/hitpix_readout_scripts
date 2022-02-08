from abc import ABC, abstractmethod
from enum import IntEnum
from dataclasses import dataclass

class Instruction(ABC):
    @abstractmethod
    def to_binary(self) -> int:
        raise NotImplementedError()

@dataclass
class Sleep(Instruction):
    def __init__(self, cycles: int) -> None:
        self.cycles = cycles
    
    def to_binary(self) -> int:
        assert (self.cycles - 1) in range(1 << 24)
        # config
        instr_mask = 0b00010001 << 24
        return instr_mask | (self.cycles - 1)

@dataclass
class Inject(Instruction):
    injection_count: int
    injection_sel: int = 0 # 0 = A, 1 = B
    
    def to_binary(self) -> int:
        assert (self.injection_count - 1) in range(1 << 16)
        assert self.injection_sel in (0, 1)
        instr_mask = 0b00010010 << 24
        instr_mask |= (self.injection_count - 1)
        instr_mask |= self.injection_sel << 16
        return instr_mask

@dataclass
class SetCfg(Instruction):
    shift_rx_invert: bool
    shift_tx_invert: bool
    shift_toggle: bool
    shift_select_dac: bool
    shift_word_len: int
    shift_clk_div: int
    shift_sample_latency: int
    
    def to_binary(self) -> int:
        assert self.shift_clk_div in range(1 << 3)
        assert (self.shift_word_len - 1) in range(1 << 5)
        assert self.shift_sample_latency in range(1 << 5)
        instr_mask = 0b00010110 << 24

        instr_mask |= self.shift_rx_invert << 23
        instr_mask |= self.shift_tx_invert << 22
        instr_mask |= self.shift_toggle << 21
        instr_mask |= self.shift_select_dac << 20
        instr_mask |= (self.shift_word_len - 1) << 15
        instr_mask |= self.shift_clk_div << 12
        instr_mask |= self.shift_sample_latency

        return instr_mask
    
    def modify(self, **kwargs) -> 'SetCfg':
        s = SetCfg(
            self.shift_rx_invert,
            self.shift_tx_invert,
            self.shift_toggle,
            self.shift_select_dac,
            self.shift_word_len,
            self.shift_clk_div,
            self.shift_sample_latency,
        )
        for k, v in kwargs.items():
            s.__setattr__(k, v)
        return s

@dataclass
class SetPins(Instruction):
    pins: int
    
    def to_binary(self) -> int:
        assert self.pins in range(1 << 24) # change when adding additional config
        return (0b00010011 << 24) | self.pins
    
    def set_pin(self, pin_number: int, value: bool) -> 'SetPins':
        assert pin_number < 24
        if value:
            return SetPins(self.pins | (1 << pin_number))
        else:
            return SetPins(self.pins & ~(1 << pin_number))

    def get_pin(self, pin_number: int) -> bool:
        return (self.pins & (1 << pin_number)) != 0

@dataclass
class ShiftIn24(Instruction):
    '''shift in 17 to 24 bits. optionally, also shift out simultaneously'''
    num_bits: int
    shift_out: bool
    data_tx: int = 0

    def to_binary(self) -> int:
        assert (self.num_bits - 17) in range(1 << 3)
        assert self.data_tx in range(1 << 24)
        instr_mask = 0b1110 << 28
        instr_mask |= (self.num_bits - 17) << 25
        instr_mask |= self.shift_out << 24
        instr_mask |= self.data_tx
        return instr_mask

@dataclass
class ShiftIn16(Instruction):
    '''shift in 1 to 16 bits. optionally, also shift out simultaneously'''
    num_bits: int
    shift_out: bool
    data_tx: int = 0
    
    def to_binary(self) -> int:
        assert (self.num_bits - 1) in range(1 << 8)
        assert self.data_tx in range(1 << 24)
        instr_mask = 0b1111000 << 25
        instr_mask |= self.shift_out << 24
        instr_mask |= (self.num_bits - 1)
        instr_mask |= self.data_tx << 8
        return instr_mask

@dataclass
class ShiftOut(Instruction):
    '''shift out up to 2**12 bits'''
    num_bits: int
    shift_out: bool
    
    def to_binary(self) -> int:
        assert (self.num_bits - 1) in range(1 << 12)
        instr_mask = 0b1111001 << 25
        instr_mask |= self.shift_out << 24
        instr_mask |= (self.num_bits - 1)
        return instr_mask

@dataclass
class Reset(Instruction):
    '''reset output shift register'''
    reset_rx: bool
    reset_tx: bool

    def to_binary(self) -> int:
        instr_mask = 0b00010100 << 24
        instr_mask |= self.reset_rx << 1
        instr_mask |= self.reset_tx << 0
        return instr_mask


@dataclass
class GetTime(Instruction):
    '''end of program'''
    def __init__(self) -> None:
        pass

    def to_binary(self) -> int:
        return 0b00010101 << 24

@dataclass
class Finish(Instruction):
    '''end of program'''
    def __init__(self) -> None:
        pass

    def to_binary(self) -> int:
        return 0

@dataclass
class Wait(Instruction):
    '''wait for external signal with id <signal_number> to read <signal_value>'''
    signal_number: int
    signal_value: bool
