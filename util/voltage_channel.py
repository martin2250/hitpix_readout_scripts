from abc import ABC, abstractmethod
import time
import json
from typing import ClassVar

class VoltageChannel(ABC):
    @abstractmethod
    def set_voltage(self, voltage: float) -> None:
        raise NotImplementedError()

    @abstractmethod
    def shutdown(self) -> None:
        raise NotImplementedError()

class ManualVoltageChannel(VoltageChannel):
    def __init__(self, name: str, _: dict) -> None:
        self._voltage = float('nan')
        self.name = name
    
    def set_voltage(self, voltage: float) -> None:
        if voltage == self._voltage:
            return
        self._voltage = voltage
        print(f'please set {self.name} to {voltage:0.2f}V')
        input('press enter to continue')
    
    def shutdown(self) -> None:
        pass

class Keithley2400VoltageChannel(VoltageChannel):
    # Keithley Menu -> Communication
    # Baud: 9600, Termination: CR+LF
    def __init__(self, name: str, config: dict) -> None:
        super().__init__()
        self.invert = config['invert']
        self.name = name
        self._voltage = float('nan')
        from pymeasure.instruments.keithley import Keithley2400
        from util.serial_adapter import SerialAdapter
        a = SerialAdapter(config['serial_port'], baudrate=9600)
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
            # raise RuntimeError(f'SMU voltage difference too large ({self.smu.voltage} V / {voltage} V)')
            pass
    
    def shutdown(self) -> None:
        self.smu.shutdown()

class HMP4040:
    devices: ClassVar[dict[str, 'HMP4040']] = {}

    def __init__(self, port: str):
        import serial
        self.serial = serial.Serial(port, 50, serial.FIVEBITS, timeout=1, xonxoff=True)
    
    def write(self, x: str) -> None:
        self.serial.write((x + '\n').encode())
        self.serial.flushOutput()
    
    def read(self) -> str:
        return self.serial.readline().decode()

class HMP4040VoltageChannel(VoltageChannel):
    def __init__(self, name: str, config: dict) -> None:
        assert config['channel'] in range(1, 5)
        self.name = name
        self.channel: int = config['channel']
        self.max_voltage: float = config['max_voltage']
        serial_port: str = config['serial_port']
        self._voltage = float('nan')

        if serial_port in HMP4040.devices:
            self.device = HMP4040.devices[serial_port]
        else:
            self.device = HMP4040(serial_port)
            HMP4040.devices[serial_port] = self.device

    def set_voltage(self, voltage: float) -> None:
        if voltage == self._voltage:
            return
        if voltage > self.max_voltage:
            raise ValueError(f'HMP4040 {self.name} {voltage=} > {self.max_voltage=}')
        self.device.write(f'INST:NSEL {self.channel}')
        self.device.write(f'VOLT {voltage}')
        time.sleep(0.5)
    
    def shutdown(self) -> None:
        pass

voltage_channels = {
    'manual': ManualVoltageChannel,
    'keithley2400': Keithley2400VoltageChannel,
    'hmp4040': HMP4040VoltageChannel,
}

def open_voltage_channel(driver: str, name: str) -> VoltageChannel:
    config = {}
    if ':' in driver:
        driver, _, cjson = driver.partition(':')
        config = json.loads(cjson)
    if not driver in voltage_channels:
        raise ValueError(f'unknown voltage channel driver: {driver}')
    return voltage_channels[driver](name, config)
