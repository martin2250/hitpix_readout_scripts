#!/usr/bin/python
from os import read
from typing import Iterable, Union, Optional
import serial
import statemachine
import time
from cobs import cobs
import threading
import queue
from dataclasses import dataclass, field
import pylibftdi

@dataclass
class Response:
    data: Optional[bytes] = None
    event: threading.Event = field(default_factory=lambda: threading.Event())

InstructionsLike = Union[list[int], list[statemachine.Instruction]]

class Readout:
    # commands
    CMD_REGISTER_READ  = 0x01
    CMD_REGISTER_WRITE = 0x02
    CMD_SM_EXEC        = 0x10
    CMD_SM_WRITE       = 0x11
    CMD_SM_START       = 0x12
    CMD_SM_ABORT       = 0x13
    CMD_HITPIX_DAC     = 0x20
    CMD_FUNCTION_CARD  = 0x30
    # registers
    ADDR_TIMER             = 0x00
    ADDR_SM_STATUS         = 0x10
    ADDR_SM_INJECTION_CTRL = 0x11
    ADDR_SM_INVERT_PINS    = 0x12

    class ReadoutError(Exception):
        # error codes
        ERROR_OK   = 0x00
        ERROR_BUSY = 0x0b
        ERROR_COMMAND = 0x0c
        ERROR_EOP  = 0x0e
        errors = {
            ERROR_OK: 'ERROR_OK',
            ERROR_BUSY: 'ERROR_BUSY',
            ERROR_COMMAND: 'ERROR_COMMAND',
            ERROR_EOP: 'ERROR_EOP',
        }
        def __init__(self, code: int) -> None:
            code_str = self.errors.get(code, f'0x{code:02X}')
            super().__init__(f'readout error code {code_str}')

    def __init__(self, serial_name: str, timeout: float = 0.5) -> None:
        self._serial_port = serial.Serial(serial_name, 3_000_000)
        self._response_queue: queue.Queue[Response] = queue.Queue()
        self._thread_read_serial = threading.Thread(target=self._read_serial, daemon=True, name='readout')
        self._thread_read_serial.start()
        self._timeout = timeout
        self._time_sync: Optional[tuple[int, float]] = None
        self._serial_port.set_low_latency_mode(True)

    def _read_serial(self):
        self._serial_port.reset_input_buffer()
        buffer = bytearray()
        while True:
            data_new = self._serial_port.read_all()
            if data_new is None:
                time.sleep(0.001)
                continue
            buffer.extend(data_new)
            # check if there are complete packets in the buffer
            if not b'\x00' in buffer:
                continue
            index = buffer.rindex(b'\x00')
            packets = buffer[:index].split(b'\x00')
            del buffer[:index+1]
            packets = [cobs.decode(packet) for packet in packets]
            for packet in packets:
                try:
                    response = self._response_queue.get(False)
                    response.data = packet
                    response.event.set()
                except queue.Empty:
                    print('received unexpected response', packet)
    
    def _expect_response(self) -> Response:
        response = Response()
        self._response_queue.put(response)
        return response

    
    def send_packet(self, packet: bytes) -> None:
        self._serial_port.write(b'\x00' + cobs.encode(packet) + b'\x00')
        self._serial_port.flushOutput()
    
    def send_packets(self, packets: Iterable[bytes]) -> None:
        data = bytearray(b'\x00')
        for packet in packets:
            data.extend(cobs.encode(packet))
            data.append(0)
        self._serial_port.write(data)
        self._serial_port.flushOutput()
    
    def write_register(self, address: int, value: int) -> None:
        assert address in range(256)
        assert value in range(1 << 32)
        self._expect_response()
        self.send_packet(bytes([self.CMD_REGISTER_WRITE, address]) + value.to_bytes(4, 'little'))
    
    def read_register(self, address: int) -> int:
        # raise NotImplementedError()
        assert address in range(256)
        response = self._expect_response()
        self.send_packet(bytes([self.CMD_REGISTER_READ, address]))
        if not response.event.wait(self._timeout):
            raise TimeoutError('no response received')
        assert response.data is not None
        assert len(response.data) == 4
        return int.from_bytes(response.data, 'little')
    
    def sm_exec(self, assembly: InstructionsLike) -> None:
        assembly_int = [
            instr if isinstance(instr, int) else instr.to_binary()
            for instr in assembly
        ]
        data = b''.join(code.to_bytes(4, 'little') for code in assembly_int)
        # write to board
        self._expect_response()
        self.send_packet(bytes([self.CMD_SM_EXEC]) + data)
    
    def sm_write(self, assembly: InstructionsLike, offset: int = 0) -> None:
        assembly_int = [
            instr if isinstance(instr, int) else instr.to_binary()
            for instr in assembly
        ]
        data = b''.join(code.to_bytes(4, 'little') for code in assembly_int)
        # write to board
        self._expect_response()
        self.send_packet(bytes([self.CMD_SM_WRITE]) + offset.to_bytes(2, 'little') + data)

    def sm_start(self, runs: int = 1) -> None:
        assert runs in range(256)
        self._expect_response()
        self.send_packet(bytes([self.CMD_SM_START, runs]))

    def sm_abort(self) -> None:
        self._expect_response()
        self.send_packet(bytes([self.CMD_SM_ABORT]))
    
    def synchronize(self) -> None:
        sync_time = time.time()
        sync_counter = self.read_register(self.ADDR_TIMER)
        self._time_sync = (sync_counter, sync_time)
    
    def convert_time(self, counter: int) -> float:
        if self._time_sync is None:
            raise Exception('time not synchronized')
        counter_sync, time_sync = self._time_sync
        counter_diff = (counter - counter_sync) & ((1 << 32) - 1) # convert to unsigned
        return time_sync + counter_diff * 1e-6
    
    def set_injection_ctrl(self, on_cycles: int, off_cycles: int) -> None:
        assert (on_cycles - 1) in range(1 << 12)
        assert (off_cycles - 1) in range(1 << 12)
        self.write_register(self.ADDR_SM_INJECTION_CTRL, (on_cycles - 1) | ((off_cycles - 1) << 16))
    
    def _write_function_card_raw(self, data: bytes) -> None:
        self._expect_response()#
        self.send_packet(bytes([self.CMD_FUNCTION_CARD]) + data)
    
    def initialize(self) -> None:
        try:
            # set all enable pins high
            self._write_function_card_raw(b'\xff')
            self.synchronize()
        except TimeoutError as err:
            msg = '| failed to initialize, is the FPGA configured? |'
            print('-' * len(msg))
            print(msg)
            print('-' * len(msg))
            raise err

    def write_function_card(self, slot_id: int, data: bytes) -> None:
        assert slot_id in range(8)
        # set CS low
        mask = (~(1 << (7 - slot_id))) & 0xff # shift register outputs are swapped on gecco
        self._write_function_card_raw(bytes([mask]))
        # write data and set CS high
        self._write_function_card_raw(data + b'\xff')

    
    @dataclass
    class SmStatus:
        remaining_runs: int
        error_count: int
        idle: bool
        active: bool

    def get_sm_status(self) -> SmStatus:
        value = self.read_register(Readout.ADDR_SM_STATUS)
        return Readout.SmStatus(
            remaining_runs = value & 0xff,
            error_count    = (value >> 8) & 0xff,
            idle           = (value & (1 << 16)) != 0,
            active         = (value & (1 << 17)) != 0,
        )
    
    def wait_sm_idle(self, timeout: float = 1.) -> None:
        t_timeout = time.monotonic() + timeout
        while not self.get_sm_status().idle:
            if time.monotonic() > t_timeout:
                raise TimeoutError('statemachine not idle')

class FastReadout:
    def __init__(self) -> None:
        self._response_queue: queue.Queue[Response] = queue.Queue()
        # open FTDI
        FT_FLOW_RTS_CTS = 0x0100
        self.ftdi = pylibftdi.Device('519166548088')
        self.ftdi.ftdi_fn.ftdi_set_bitmode(0xff, 0x00)
        time.sleep(10e-3)
        self.ftdi.ftdi_fn.ftdi_set_bitmode(0xff, 0x40)
        self.ftdi.ftdi_fn.ftdi_setflowctrl(FT_FLOW_RTS_CTS, 0, 0)

        # start RX thread
        threading.Thread(target=self.read, daemon=True, name='fastreadout').start()
    
    def read(self) -> None:
        # clear buffer
        while self.ftdi.read(16*4096):
            pass
        buffer = bytearray()
        while True:
            data_new = self.ftdi.read(16*4096)
            if not isinstance(data_new, bytes) or not data_new:
                time.sleep(0.001)
                continue
            buffer.extend(data_new)
            # check if there are complete packets in the buffer
            if not b'\x00' in buffer:
                continue
            index = buffer.rindex(b'\x00')
            packets = buffer[:index].split(b'\x00')
            del buffer[:index+1]
            for packet in packets:
                try:
                    response = self._response_queue.get(False)
                    response.data = cobs.decode(packet)
                    response.event.set()
                except queue.Empty:
                    print('received unexpected response', packet)
    
    def expect_response(self) -> Response:
        response = Response()
        self._response_queue.put(response)
        return response

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
