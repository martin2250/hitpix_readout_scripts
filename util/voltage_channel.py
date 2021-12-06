from abc import ABC, abstractmethod
import time

from util.configuration import ReadoutBoardConfig

class VoltageChannel(ABC):
    @abstractmethod
    def set_voltage(self, voltage: float) -> None:
        raise NotImplementedError()

    @abstractmethod
    def shutdown(self) -> None:
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
    
    def shutdown(self) -> None:
        pass

class Keithley2400VoltageChannel(VoltageChannel):
    # Keithley Menu -> Communication
    # Baud: 9600, Termination: CR+LF
    def __init__(self, serial_port: str, invert=True) -> None:
        super().__init__()
        self.invert = invert
        self._voltage = float('nan')
        from pymeasure.instruments.keithley import Keithley2400
        from util.serial_adapter import SerialAdapter
        a = SerialAdapter(serial_port, baudrate=9600)
        self.smu = Keithley2400(a)
        self.smu.shutdown()
        self.smu.reset()
        time.sleep(0.1)
        self.smu.apply_voltage(compliance_current=120e-6)
        self.smu.measure_voltage()
        self.smu.auto_range_source()
    
    def set_voltage(self, voltage: float) -> None:
        if self.invert:
            voltage = -voltage
        if voltage == self._voltage:
            return
        self._voltage = voltage
        # send to SMU
        self.smu.source_voltage = voltage
        self.smu.enable_source()
        for _ in range(5):
            time.sleep(0.05)
            voltage_readback = self.smu.voltage
            if abs(voltage - voltage_readback) < 0.1:
                break
        else:
            raise RuntimeError(f'SMU voltage difference too large ({self.smu.voltage} V / {voltage} V)')
    
    def shutdown(self) -> None:
        self.smu.shutdown()

def open_voltage_channel(name: str, board_config: ReadoutBoardConfig) -> VoltageChannel:
    if name == 'manual':
        return ManualVoltageChannel()
    if name == 'keithley2400':
        return Keithley2400VoltageChannel(board_config.hv_smu_serial_port)
    else:
        raise ValueError(f'unknown voltage channel type: {name}')
