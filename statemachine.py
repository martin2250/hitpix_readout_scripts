#!/usr/bin/python
from abc import ABC
from dataclasses import dataclass

class Instruction(ABC):
    pass

class Sleep(Instruction):
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

class Inject(Instruction):
    def __init__(self, injection_count: int) -> None:
        self.injection_count = injection_count

class SetPins(Instruction):
    def __init__(self, **kwargs: bool) -> None:
        self.pins = kwargs

class ShiftIn24(Instruction):
    '''shift in 17 to 24 bits. optionally, also shift out simultaneously'''
    def __init__(self, num_bits: int, shift_out: bool, data_in: int = 0) -> None:
        assert num_bits in range(17, 25)
        assert data_in in range(1 << 24)
        self.data_in = data_in
        self.num_bits = num_bits
        self.shift_out = shift_out

class ShiftIn16(Instruction):
    '''shift in 1 to 16 bits. optionally, also shift out simultaneously'''
    def __init__(self, num_bits: int, shift_out: bool, data_in: int = 0) -> None:
        assert (num_bits - 1) in range(1 << 8)
        assert data_in in range(1 << 24)
        self.data_in = data_in
        self.num_bits = num_bits
        self.shift_out = shift_out

class ShiftOut(Instruction):
    '''shift in 1 to 16 bits. optionally, also shift out simultaneously'''
    def __init__(self, num_bits: int, shift_out: bool) -> None:
        assert (num_bits - 1) in range(1 << 12)
        self.num_bits = num_bits
        self.shift_out = shift_out

class Repeat(Instruction):
    '''repeat next instruction <repeat_count> times'''
    def __init__(self, repeat_count: int) -> None:
        self.repeat_count = repeat_count

class Reset(Instruction):
    '''reset output shift register'''
    def __init__(self, reset_rx: bool, reset_tx: bool) -> None:
        self.reset_rx = reset_rx
        self.reset_tx = reset_tx

class Wait(Instruction):
    '''wait for external signal with id <signal_number> to read <signal_value>'''
    def __init__(self, signal_number: int, signal_value: bool) -> None:
        self.signal_number = signal_number
        self.signal_value = signal_value

class Finish(Instruction):
    '''end of program'''
    def __init__(self) -> None:
        pass


@dataclass
class Pin:
    bit_position: int
    invert: bool
    state: bool = False

class Statemachine:
    def __init__(self, pins: dict[str, Pin], frequency: float):
        self.pins = pins
        self.frequency = frequency
    
    def compile(self, instructions: list[Instruction]) -> list[int]:
        program: list[int] = []

        for instruction in instructions:
            if isinstance(instruction, Sleep):
                assert instruction.seconds > 0
                cycles = int(instruction.seconds * self.frequency)
                # config
                instr_mask = 0b00010001 << 24
                instr_count_max = (1 << 24) - 1
                # divide up cycles
                while cycles > 0:
                    cycles_this_round = min(cycles, instr_count_max)
                    cycles -= cycles_this_round
                    program.append(instr_mask | cycles_this_round)
            elif isinstance(instruction, Inject):
                assert instruction.injection_count in range(1 << 12)
                instr_mask = 0b00010010 << 24
                instr_mask |= (instruction.injection_count - 1)
                program.append(instr_mask)
            elif isinstance(instruction, SetPins):
                for name, state_new in instruction.pins.items():
                    assert name in self.pins
                    self.pins[name].state = state_new
                instr_mask = 0b00010011 << 24
                for pin in self.pins.values():
                    if pin.state != pin.invert:
                        instr_mask |= 1 << pin.bit_position
                program.append(instr_mask)
            elif isinstance(instruction, ShiftIn24):
                instr_mask = 0b1110 << 28
                instr_mask |= (instruction.num_bits - 17) << 25
                instr_mask |= instruction.shift_out << 24
                instr_mask |= instruction.data_in
                program.append(instr_mask)
            elif isinstance(instruction, ShiftIn16):
                instr_mask = 0b1111000 << 25
                instr_mask |= instruction.shift_out << 24
                instr_mask |= (instruction.num_bits - 1)
                instr_mask |= instruction.data_in << 8
                program.append(instr_mask)
            elif isinstance(instruction, ShiftOut):
                instr_mask = 0b1111001 << 25
                instr_mask |= instruction.shift_out << 24
                instr_mask |= (instruction.num_bits - 1)
                program.append(instr_mask)
            elif isinstance(instruction, Finish):
                program.append(0)
            elif isinstance(instruction, Reset):
                instr_mask = 0b00010100 << 24
                instr_mask |= instruction.reset_rx << 1
                instr_mask |= instruction.reset_tx << 0
                program.append(instr_mask)
            elif isinstance(instruction, Repeat):
                assert (instruction.repeat_count - 1) in range(1 << 12)
                instr_mask = 0b00010000 << 24
                instr_mask |= (instruction.repeat_count - 1)
                program.append(instr_mask)
            else:
                raise ValueError(f'unknown instruction type {type(instruction)}')

        return program

    @staticmethod
    def get_statemachine_hitpix1() -> 'Statemachine':
        return Statemachine(
            pins={
                'ro_ldconfig': Pin(0, False),
                'ro_psel':     Pin(1, False),
                'ro_penable':  Pin(2, False),
                'ro_ldcnt':    Pin(3, False),
                'ro_rescnt':   Pin(4, False),
                'ro_frame':    Pin(5, False),
            },
            frequency=100e6,
        )

if __name__ == '__main__':
    sm = Statemachine.get_statemachine_hitpix1()
