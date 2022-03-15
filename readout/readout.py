import math
import queue
import struct
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Union

import serial
from cobs import cobs
from util.time_sync import TimeSync

from . import Response, instructions

InstructionsLike = Union[list[int], list[instructions.Instruction]]

class PacketComm:
    def __init__(self):
        self.callback: Optional[Callable[[bytes], None]] = None
    
    def send_packet(self, packet: bytes) -> None:
        raise NotImplementedError()
    
    def send_packets(self, packets: Iterable[bytes]) -> None:
        for p in packets:
            self.send_packet(p)
    
    def close(self) -> None:
        pass

class SerialCobsComm(PacketComm):
    def __init__(self, serial_name: str):
        super().__init__()
        self.event_stop = threading.Event()
        self._serial_port = serial.Serial(serial_name, 3_000_000)
        self._serial_port.set_low_latency_mode(True)
        self._thread_read_serial = threading.Thread(
            target=self._read_serial,
            daemon=True,
            name='readout',
        )
        self._thread_read_serial.start()

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
                if self.callback is not None:
                    self.callback(packet)
    
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

class Readout:
    # commands
    CMD_REGISTER_READ = 0x01
    CMD_REGISTER_WRITE = 0x02
    CMD_SM_EXEC = 0x10
    CMD_SM_WRITE = 0x11
    CMD_SM_START = 0x12
    CMD_SM_ABORT = 0x13
    CMD_SM_SOFT_ABORT = 0x15
    CMD_FAST_TX_FLUSH = 0x14
    CMD_HITPIX_DAC = 0x20
    CMD_FUNCTION_CARD = 0x30
    # registers
    ADDR_TIMER = 0x00
    ADDR_SM_STATUS = 0x10
    ADDR_SM_INJECTION_CTRL = 0x11
    ADDR_SM_INVERT_PINS = 0x12
    ADDR_MMCM_CONFIG_0 = 0x20
    ADDR_MMCM_CONFIG_1 = 0x21
    ADDR_MMCM_CONFIG_2 = 0x22
    ADDR_RO_CLKS_DIV1 = 0x30
    ADDR_VERSION = 0xf0
    ADDR_CTR_COMM_RX = 0xd0
    ADDR_CTR_COMM_TX = 0xd1
    ADDR_CTR_FAST_TX = 0xd2

    class ReadoutError(Exception):
        # error codes
        ERROR_OK = 0x00
        ERROR_BUSY = 0x0b
        ERROR_COMMAND = 0x0c
        ERROR_EOP = 0x0e
        errors = {
            ERROR_OK: 'ERROR_OK',
            ERROR_BUSY: 'ERROR_BUSY',
            ERROR_COMMAND: 'ERROR_COMMAND',
            ERROR_EOP: 'ERROR_EOP',
        }

        def __init__(self, code: int) -> None:
            code_str = self.errors.get(code, f'0x{code:02X}')
            super().__init__(f'readout error code {code_str}')

    def __init__(self, comm: PacketComm, timeout: float = 1.5) -> None:
        self.comm = comm
        self.comm.callback = self.receive_callback
        self._response_queue: queue.Queue[Response] = queue.Queue()
        self._timeout = timeout
        self._sm_error_count = 0
        self._sm_prog_bits = 12
        self.frequency_mhz = float('nan')
        self.frequency_mhz_set = float('nan')
        self.debug_responses = False

    def close(self) -> None:
        self.comm.close()
    
    def receive_callback(self, packet: bytes) -> None:
        try:
            response = self._response_queue.get(False)
            if self.debug_responses:
                print(response.name, '>>>', packet[:8].hex())
            response.data = packet
            response.event.set()
        except queue.Empty:
            print('readout received unexpected response', packet)

    def _expect_response(self, name: Optional[str] = None) -> Response:
        if (not name) and self.debug_responses:
            lines = '\n'.join(traceback.format_stack())
            stack = []
            for s in lines.splitlines():
                s = s.strip()
                if not s or s.startswith('File'):
                    continue
                # remove unittest call stack
                if s == 'method()':
                    stack = []
                    continue
                # remove common stuff
                if '_expect_response()' in s:
                    break
                stack.append(s)
            name = '::'.join(stack)
        response = Response(name=name)
        self._response_queue.put(response)
        return response

    def write_register(self, address: int, value: int) -> None:
        assert address in range(256)
        assert value in range(1 << 32)
        resp = self._expect_response()
        self.comm.send_packet(
            bytes([self.CMD_REGISTER_WRITE, address]) + value.to_bytes(4, 'little'))
        assert resp.event.wait()

    def read_register(self, address: int) -> int:
        assert address in range(256)
        response = self._expect_response()
        self.comm.send_packet(bytes([self.CMD_REGISTER_READ, address]))
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
        resp = self._expect_response()
        self.comm.send_packet(bytes([self.CMD_SM_EXEC]) + data)
        assert resp.event.wait()

    def sm_write(self, assembly: InstructionsLike, offset: int = 0) -> int:
        '''returns offset of next instruction'''
        assembly_int = [
            instr if isinstance(instr, int) else instr.to_binary()
            for instr in assembly
        ]
        assert (len(assembly_int) + offset) <= (1 <<
                                                self._sm_prog_bits), 'sm prog too large'
        data = b''.join(code.to_bytes(4, 'little') for code in assembly_int)
        # write to board
        resp = self._expect_response()
        self.comm.send_packet(bytes([self.CMD_SM_WRITE]) +
                         offset.to_bytes(2, 'little') + data)
        assert resp.event.wait()
        return offset + len(assembly_int)

    def sm_start(self, runs: int = 1, offset: int = 0, packets: int = 1) -> None:
        """_summary_

        Args:
            runs (int, optional): _description_. Defaults to 1.
            offset (int, optional): _description_. Defaults to 0.
            packets (int, optional): 0 == infinite. Defaults to 1.
        """
        assert runs in range(0x10000)
        assert offset in range(0x10000)
        assert packets in range(0x10000)
        resp = self._expect_response()
        self.comm.send_packet(struct.pack(
            '<BHHH',
            self.CMD_SM_START,
            offset,
            runs,
            (packets - 1) & 0xffff,
        ))
        assert resp.event.wait()

    def sm_abort(self) -> None:
        resp = self._expect_response()
        self.comm.send_packet(bytes([self.CMD_SM_ABORT]))
        assert resp.event.wait()

    def sm_soft_abort(self) -> None:
        resp = self._expect_response()
        self.comm.send_packet(bytes([self.CMD_SM_SOFT_ABORT]))
        assert resp.event.wait()

    def fast_tx_flush(self) -> None:
        resp = self._expect_response()
        self.comm.send_packet(bytes([self.CMD_FAST_TX_FLUSH]))
        assert resp.event.wait()

    def get_synchronization(self) -> TimeSync:
        sync_time = time.time()
        sync_counter = self.read_register(self.ADDR_TIMER)
        return TimeSync(sync_counter, sync_time, 1e-6)

    def set_injection_ctrl(self, on_cycles: int, off_cycles: int) -> None:
        assert (on_cycles - 1) in range(1 << 16)
        assert (off_cycles - 1) in range(1 << 16)
        self.write_register(self.ADDR_SM_INJECTION_CTRL,
                            (on_cycles - 1) | ((off_cycles - 1) << 16))

    def _write_function_card_raw(self, data: bytes) -> None:
        resp = self._expect_response()
        self.comm.send_packet(bytes([self.CMD_FUNCTION_CARD]) + data)
        assert resp.event.wait()

    def set_readout_clock_sequence(self, clk1: int, clk2: int) -> None:
        assert clk1 in range(1 << 12)
        assert clk2 in range(1 << 12)
        self.write_register(self.ADDR_RO_CLKS_DIV1, (clk2 << 16) | clk1)

    def initialize(self) -> None:
        try:
            # set all enable pins high
            self.fast_tx_flush()
            self._write_function_card_raw(b'\xff')
            # check version
            version = self.get_version()
            # supported versions
            if version.readout not in [0x015, 0x016]:
                raise RuntimeError(
                    f'unsupported readout version 0x{version.readout:04X}')
            # sm prog
            if version.readout >= 0x0010:
                self._sm_prog_bits = 14
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

    def write_function_card(self, slot_id: int, data: bytes) -> None:
        assert slot_id in range(8)
        # set CS low
        # shift register outputs are swapped on gecco
        mask = (~(1 << (7 - slot_id))) & 0xff
        packets = [
            # enable cs line for this slot
            bytes([self.CMD_FUNCTION_CARD, mask]),
            bytes([self.CMD_FUNCTION_CARD]) + data + \
            b'\xff',  # write data and set CS high
        ]
        resp = [self._expect_response() for _ in packets]
        self.comm.send_packets(packets)
        for r in resp:
            assert r.event.wait()

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
            remaining_runs=value & 0xffff,
            error_count=error_count,
            idle=(value & (1 << 24)) != 0,
            active=(value & (1 << 25)) != 0,
        )

    def wait_sm_idle(self, timeout: float = 1.) -> None:
        t_timeout = time.monotonic() + timeout
        while self.get_sm_status().active:
            if time.monotonic() > t_timeout:
                raise TimeoutError('statemachine not idle')

    def set_system_clock(self, frequency_mhz: float, dry_run: bool = False) -> float:
        '''set system clock frequency to nearest possible value
        param frequency_mhz: bit rate when using lowest divider setting
        '''
        assert frequency_mhz <= 190.0
        assert frequency_mhz >= 15.0
        from util.xilinx import pll7series

        # find values for 4x bitrate (output divider is doubled in FPGA)
        div_fb, div_out, freq_gen = pll7series.optimize_vco_and_divider(
            100.0, 6 * frequency_mhz)
        if dry_run:
            return freq_gen / 6
        regs = pll7series.get_register_values(div_fb, div_out, 'optimized')
        for i, val in enumerate(regs):
            self.write_register(self.ADDR_MMCM_CONFIG_0 + i, val)
        # suppress responses
        self._expect_response('dummy')
        self._expect_response('dummy')
        # wait for hardware
        time.sleep(0.8)
        try:
            while True:
                self._response_queue.get_nowait()
        except queue.Empty:
            pass
        # update system frequency
        self.frequency_mhz = freq_gen / 6
        self.frequency_mhz_set = frequency_mhz
        # re-initialize
        self.initialize()
        return self.frequency_mhz

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

    ############################################################################

    @dataclass
    class CommCounters:
        comm_rx_packets: int
        comm_rx_bytes: int
        comm_tx_packets: int
        comm_tx_bytes: int
        fast_tx_packets: int
        fast_tx_bytes: int

    def get_comm_counters(self) -> CommCounters:
        comm_rx = self.read_register(self.ADDR_CTR_COMM_RX)
        comm_tx = self.read_register(self.ADDR_CTR_COMM_TX)
        fast_tx = self.read_register(self.ADDR_CTR_FAST_TX)
        return self.CommCounters(
            comm_rx_packets = (comm_rx >> 16) & 0xffff,
            comm_rx_bytes = comm_rx & 0xffff,
            comm_tx_packets = (comm_tx >> 16) & 0xffff,
            comm_tx_bytes = comm_tx & 0xffff,
            fast_tx_packets = (fast_tx >> 16) & 0xffff,
            fast_tx_bytes = fast_tx & 0xffff,
        )
