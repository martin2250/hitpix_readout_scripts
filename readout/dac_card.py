from .readout import Readout

class DacCard:
    def __init__(self, slot_id: int, num_dacs: int, voltage_supply: float, voltage_max: float, readout: Readout) -> None:
        assert voltage_max <= voltage_supply
        self.slot_id = slot_id
        self.voltage_supply = voltage_supply
        self.voltage_max = voltage_max
        self.num_dacs = num_dacs
        self.voltages = [0.0 for _ in range(num_dacs)]
        self.readout = readout
    
    def update(self) -> None:
        '''voltage needs about 3ms to stabilize (almost linear ramp up)'''
        code_max = (1 << 14) - 1
        data = bytearray()
        for i, voltage in enumerate(reversed(self.voltages)):
            # check voltage is safe
            if not (0.0 <= voltage <= self.voltage_max):
                raise ValueError(f'voltage #{i} value {voltage:0.2f} is out of range!')
            # convert to 16 bit code
            code = int(code_max * voltage / self.voltage_supply)
            # add code to data array
            data.extend((code << 2).to_bytes(2, 'big'))
        # write to board
        self.readout.write_function_card(self.slot_id, data)

    def set_voltage(self, output_id: int, voltage: float, immediate: bool = True) -> None:
        self.voltages[output_id] = voltage
        if immediate:
            self.update()
