from tracemalloc import start
from readout.readout import InstructionsLike, Readout


class SmMultiprog:
    def __init__(self, ro: Readout, start_offset: int = 0):
        self.ro = ro
        self.offset = start_offset
    
    def add_program(self, assembly: InstructionsLike) -> int:
        start_offset = self.offset
        self.offset = self.ro.sm_write(assembly, start_offset)
        return start_offset
