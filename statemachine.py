#!/usr/bin/python
from abc import ABC, abstractmethod
from enum import IntEnum

class Instruction(ABC):
    @abstractmethod
    def to_binary(self) -> int:
        raise NotImplementedError()

class Sleep(Instruction):
    def __init__(self, cycles: int) -> None:
        self.cycles = cycles
    
    def to_binary(self) -> int:
        assert (self.cycles - 1) in range(1 << 24)
        # config
        instr_mask = 0b00010001 << 24
        return instr_mask | (self.cycles - 1)

class Inject(Instruction):
    def __init__(self, injection_count: int) -> None:
        self.injection_count = injection_count
    
    def to_binary(self) -> int:
        assert (self.injection_count - 1) in range(1 << 12)
        instr_mask = 0b00010010 << 24
        instr_mask |= (self.injection_count - 1)
        return instr_mask

class SetCfg(Instruction):
    def __init__(self, shift_rx_invert: bool, shift_tx_invert: bool, shift_toggle: bool, shift_select_dac: bool, shift_word_len: int, shift_clk_div: int, pins: int) -> None:
        self.shift_rx_invert = shift_rx_invert
        self.shift_tx_invert = shift_tx_invert
        self.shift_toggle = shift_toggle
        self.shift_select_dac = shift_select_dac
        self.shift_word_len = shift_word_len
        self.shift_clk_div = shift_clk_div
        self.pins = pins
    
    def to_binary(self) -> int:
        assert self.shift_clk_div in range(1 << 3)
        assert (self.shift_word_len - 1) in range(1 << 5)
        assert self.pins in range(1 << 12) # change when adding additional config
        instr_mask = 0b00010011 << 24

        instr_mask |= self.shift_rx_invert << 23
        instr_mask |= self.shift_tx_invert << 22
        instr_mask |= self.shift_toggle << 21
        instr_mask |= self.shift_select_dac << 20
        instr_mask |= (self.shift_word_len - 1) << 15
        instr_mask |= self.shift_clk_div << 12
        instr_mask |= self.pins

        return instr_mask
    
    def modify(self, **kwargs) -> 'SetCfg':
        s = SetCfg(
            self.shift_rx_invert,
            self.shift_tx_invert,
            self.shift_toggle,
            self.shift_select_dac,
            self.shift_word_len,
            self.shift_clk_div,
            self.pins,
        )
        for k, v in kwargs.items():
            s.__setattr__(k, v)
        return s

    def set_pin(self, pin_number: int, value: bool) -> 'SetCfg':
        assert pin_number < 12 
        if value:
            return self.modify(pins=(self.pins | (1 << pin_number)))
        else:
            return self.modify(pins=(self.pins & ~(1 << pin_number)))

class ShiftIn24(Instruction):
    '''shift in 17 to 24 bits. optionally, also shift out simultaneously'''
    def __init__(self, num_bits: int, shift_out: bool, data_tx: int = 0) -> None:
        self.data_tx = data_tx
        self.num_bits = num_bits
        self.shift_out = shift_out
    
    def to_binary(self) -> int:
        assert (self.num_bits - 17) in range(1 << 3)
        assert self.data_tx in range(1 << 24)
        instr_mask = 0b1110 << 28
        instr_mask |= (self.num_bits - 17) << 25
        instr_mask |= self.shift_out << 24
        instr_mask |= self.data_tx
        return instr_mask

class ShiftIn16(Instruction):
    '''shift in 1 to 16 bits. optionally, also shift out simultaneously'''
    def __init__(self, num_bits: int, shift_out: bool, data_tx: int = 0) -> None:
        self.data_tx = data_tx
        self.num_bits = num_bits
        self.shift_out = shift_out
    
    def to_binary(self) -> int:
        assert (self.num_bits - 1) in range(1 << 8)
        assert self.data_tx in range(1 << 24)
        instr_mask = 0b1111000 << 25
        instr_mask |= self.shift_out << 24
        instr_mask |= (self.num_bits - 1)
        instr_mask |= self.data_tx << 8
        return instr_mask

class ShiftOut(Instruction):
    '''shift in 1 to 16 bits. optionally, also shift out simultaneously'''
    def __init__(self, num_bits: int, shift_out: bool) -> None:
        self.num_bits = num_bits
        self.shift_out = shift_out
    
    def to_binary(self) -> int:
        assert (self.num_bits - 1) in range(1 << 12)
        instr_mask = 0b1111001 << 25
        instr_mask |= self.shift_out << 24
        instr_mask |= (self.num_bits - 1)
        return instr_mask

class Repeat(Instruction):
    '''repeat next instruction <repeat_count> times'''
    def __init__(self, repeat_count: int) -> None:
        self.repeat_count = repeat_count
    
    def to_binary(self) -> int:
        assert (self.repeat_count - 1) in range(1 << 12)
        instr_mask = 0b00010000 << 24
        instr_mask |= (self.repeat_count - 1)
        return instr_mask

class Reset(Instruction):
    '''reset output shift register'''
    def __init__(self, reset_rx: bool, reset_tx: bool) -> None:
        self.reset_rx = reset_rx
        self.reset_tx = reset_tx
    
    def to_binary(self) -> int:
        instr_mask = 0b00010100 << 24
        instr_mask |= self.reset_rx << 1
        instr_mask |= self.reset_tx << 0
        return instr_mask


class GetTime(Instruction):
    '''end of program'''
    def __init__(self) -> None:
        pass

    def to_binary(self) -> int:
        return 0b00010101 << 24

class Finish(Instruction):
    '''end of program'''
    def __init__(self) -> None:
        pass

    def to_binary(self) -> int:
        return 0

class Wait(Instruction):
    '''wait for external signal with id <signal_number> to read <signal_value>'''
    def __init__(self, signal_number: int, signal_value: bool) -> None:
        self.signal_number = signal_number
        self.signal_value = signal_value
        raise NotImplementedError()


class HitPix1Pins(IntEnum):
    ro_ldconfig = 0
    ro_psel     = 1
    ro_penable  = 2
    ro_ldcnt    = 3
    ro_rescnt   = 4
    ro_frame    = 5
    dac_ld      = 6