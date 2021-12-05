from abc import ABC, abstractmethod
import time

class VoltageChannel(ABC):
    @abstractmethod
    def set_voltage(self, voltage: float) -> None:
        raise NotImplementedError()

class ManualVoltageChannel(VoltageChannel):
    def __init__(self) -> None:
        self._voltage = float('nan')
    
    def set_voltage(self, voltage: float) -> None:
        if voltage == self._voltage:
            return
        self._voltage = voltage
        print(f'please set HV to {voltage:0.2f}V')
        input('press enter to continue')

class Keithley2400VoltageChannel(VoltageChannel):
    def __init__(self) -> None:
        super().__init__()
        self._voltage = float('nan')
        from pymeasure.instruments.keithley import Keithley2400
        self.smu = Keithley2400('')
        self.smu.shutdown()
        self.smu.reset()
        time.sleep(0.1)
        self.smu.apply_voltage(compliance_current=100e-6)
    
    def set_voltage(self, voltage: float) -> None:
        if voltage == self._voltage:
            return
        self._voltage = voltage
        self.smu.source_voltage = voltage
        time.sleep(0.1)

def open_voltage_channel(name: str) -> VoltageChannel:
    if name == 'manual':
        return ManualVoltageChannel()
    else:
        raise ValueError(f'unknown voltage channel type: {name}')
