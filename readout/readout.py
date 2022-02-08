import math
from typing import Any, Iterable, Union, Optional
import serial
from . import instructions
import time
from cobs import cobs
import threading
import queue
from dataclasses import dataclass
import struct

from . import Response

InstructionsLike = Union[list[int], list[instructions.Instruction]]

class Readout:
    # commands
    CMD_REGISTER_READ  = 0x01
    CMD_REGISTER_WRITE = 0x02
    CMD_SM_EXEC        = 0x10
    CMD_SM_WRITE       = 0x11
    CMD_SM_START       = 0x12
    CMD_SM_ABORT       = 0x13
    CMD_FAST_TX_FLUSH  = 0x14
    CMD_HITPIX_DAC     = 0x20
    CMD_FUNCTION_CARD  = 0x30
    # registers
    ADDR_TIMER             = 0x00
    ADDR_SM_STATUS         = 0x10
    ADDR_SM_INJECTION_CTRL = 0x11
    ADDR_SM_INVERT_PINS    = 0x12
    ADDR_MMCM_CONFIG       = 0x20
    ADDR_VERSION           = 0xf0

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
        self.event_stop = threading.Event()
        self._thread_read_serial = threading.Thread(target=self._read_serial, daemon=True, name='readout')
        self._thread_read_serial.start()
        self._timeout = timeout
        self._time_sync: Optional[tuple[int, float]] = None
        self._serial_port.set_low_latency_mode(True)
        self._sm_error_count = 0
        self._sm_prog_bits = 12
        self.frequency_mhz = float('nan')
        self.frequency_vco_mhz = float('nan')
    
    def close(self) -> None:
        self.event_stop.set()
        self._thread_read_serial.join()
        self._serial_port.close()

    def _read_serial(self):
        self._serial_port.reset_input_buffer()
        buffer = bytearray()
        while not self.event_stop.is_set():
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
                    print('readout received unexpected response', packet)
    
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
    
    def sm_write(self, assembly: InstructionsLike, offset: int = 0) -> int:
        '''returns offset of next instruction'''
        assembly_int = [
            instr if isinstance(instr, int) else instr.to_binary()
            for instr in assembly
        ]
        assert (len(assembly_int) + offset) <= (1 << self._sm_prog_bits), 'sm prog too large'
        data = b''.join(code.to_bytes(4, 'little') for code in assembly_int)
        # write to board
        self._expect_response()
        self.send_packet(bytes([self.CMD_SM_WRITE]) + offset.to_bytes(2, 'little') + data)
        return offset + len(assembly_int)

    def sm_start(self, runs: int = 1, offset: int = 0) -> None:
        assert runs in range(0x10000)
        assert offset in range(0x10000)
        self._expect_response()
        self.send_packet(struct.pack('<BHH', self.CMD_SM_START, offset, runs))

    def sm_abort(self) -> None:
        self._expect_response()
        self.send_packet(bytes([self.CMD_SM_ABORT]))

    def fast_tx_flush(self) -> None:
        self._expect_response()
        self.send_packet(bytes([self.CMD_FAST_TX_FLUSH]))
    
    def synchronize(self) -> None:
        sync_time = time.time()
        sync_counter = self.read_register(self.ADDR_TIMER)
        self._time_sync = (sync_counter, sync_time)
    
    def convert_time(self, counter: Any) -> Any:
        # use Any type to allow ints and ndarrays
        if self._time_sync is None:
            raise Exception('time not synchronized')
        counter_sync, time_sync = self._time_sync
        counter_diff = (counter - counter_sync) & ((1 << 32) - 1) # convert to unsigned
        return time_sync + counter_diff * 1e-6
    
    def set_injection_ctrl(self, on_cycles: int, off_cycles: int) -> None:
        assert (on_cycles - 1) in range(1 << 16)
        assert (off_cycles - 1) in range(1 << 16)
        self.write_register(self.ADDR_SM_INJECTION_CTRL, (on_cycles - 1) | ((off_cycles - 1) << 16))
    
    def _write_function_card_raw(self, data: bytes) -> None:
        self._expect_response()
        self.send_packet(bytes([self.CMD_FUNCTION_CARD]) + data)
    
    def initialize(self) -> None:
        try:
            # set all enable pins high
            self.fast_tx_flush()
            self._write_function_card_raw(b'\xff')
            self.synchronize()
            # check version
            version = self.get_version()
            # supported versions
            if version.readout not in [0x008, 0x009, 0x010, 0x012]:
                raise RuntimeError(f'unsupported readout version 0x{version.readout:04X}')
            # sm prog
            if version.readout >= 0x0010:
                self._sm_prog_bits = 14
            # frequency
            if version.readout < 0x0010:
                self.frequency_mhz = 200
            elif version.readout == 0x0010:
                self.frequency_mhz = 100
            else:
                self.frequency_vco_mhz = 1200.0
                if math.isnan(self.frequency_mhz):
                    self.frequency_mhz = 150.0
            # check statemachine running
            if not self.get_sm_status().idle:
                print('init: state machine was still running, aborting')
                self.sm_abort()
        except TimeoutError as err:
            msg = '| failed to initialize, FPGA configured and FTDI 60MHz active? |'
            print('-' * len(msg))
            print(msg)
            print('-' * len(msg))
            raise err

    # # slow version (two calls to send_packet)
    # def write_function_card(self, slot_id: int, data: bytes) -> None:
    #     assert slot_id in range(8)
    #     # set CS low
    #     mask = (~(1 << (7 - slot_id))) & 0xff # shift register outputs are swapped on gecco
    #     self._write_function_card_raw(bytes([mask]))
    #     # write data and set CS high
    #     self._write_function_card_raw(data + b'\xff')

    def write_function_card(self, slot_id: int, data: bytes) -> None:
        assert slot_id in range(8)
        # set CS low
        mask = (~(1 << (7 - slot_id))) & 0xff # shift register outputs are swapped on gecco
        packets = [
            bytes([self.CMD_FUNCTION_CARD, mask]), # enable cs line for this slot
            bytes([self.CMD_FUNCTION_CARD]) + data + b'\xff', # write data and set CS high
        ]
        for _ in packets:
            self._expect_response()
        self.send_packets(packets)
    
    @dataclass
    class SmStatus:
        remaining_runs: int
        error_count: int
        idle: bool
        active: bool

    def get_sm_status(self) -> SmStatus:
        value = self.read_register(Readout.ADDR_SM_STATUS)
        error_count = (value >> 16) & 0xff
        if error_count != self._sm_error_count:
            print(f'SM errors: {error_count}')
            self._sm_error_count = error_count
        return Readout.SmStatus(
            remaining_runs = value & 0xffff,
            error_count    = error_count,
            idle           = (value & (1 << 24)) != 0,
            active         = (value & (1 << 25)) != 0,
        )
    
    def wait_sm_idle(self, timeout: float = 1.) -> None:
        t_timeout = time.monotonic() + timeout
        while self.get_sm_status().active:
            if time.monotonic() > t_timeout:
                raise TimeoutError('statemachine not idle')
    
    def set_system_clock(self, frequency_mhz: float):
        '''set system clock frequency to nearest possible value, rounds down'''
        assert frequency_mhz < 150.0
        assert math.isfinite(self.frequency_vco_mhz)
        # calculate values
        divider = 2 * int(math.ceil(self.frequency_vco_mhz / frequency_mhz / 4))
        # write values to register
        assert divider in range(1 << 6)
        value = (divider << 6) | divider
        self.write_register(self.ADDR_MMCM_CONFIG, value)
        # update system frequency
        self.frequency_mhz = self.frequency_vco_mhz / (2 * divider)
        print(f'readout: update readout frequency {2*divider=} f={self.frequency_mhz:0.2f}MHz')
        # wait for hardware and re-initialize
        time.sleep(0.3)
        self.fast_tx_flush()
        self.initialize()
    
    ############################################################################

    @dataclass
    class VersionInfo:
        chip: int
        adapter: int
        readout: int

    def get_version(self) -> VersionInfo:
        raw = self.read_register(self.ADDR_VERSION)
        return self.VersionInfo(
            chip=(raw >> 24) & 0xff,
            adapter=(raw >> 16) & 0xff,
            readout=(raw >> 0) & 0xffff,
        )
