import queue
import threading
import time

from cobs import cobs
import pylibftdi

from . import Response


class FastReadout:
    def __init__(self, serial_number: str) -> None:
        self.serial_number = serial_number
        self._response_queue: queue.Queue[Response] = queue.Queue()
        # open FTDI
        FT_FLOW_RTS_CTS = 0x0100
        self.ftdi = pylibftdi.Device(self.serial_number)
        self.ftdi.ftdi_fn.ftdi_usb_purge_rx_buffer()
        self.ftdi.ftdi_fn.ftdi_set_bitmode(0xff, 0x00)
        time.sleep(10e-3)
        self.ftdi.ftdi_fn.ftdi_set_bitmode(0xff, 0x40)
        self.ftdi.ftdi_fn.ftdi_setflowctrl(FT_FLOW_RTS_CTS, 0, 0)
        self.ftdi.ftdi_fn.ftdi_set_latency_timer(2)
        self.ftdi.ftdi_fn.ftdi_read_data_set_chunksize(0x10000)
        for _ in range(3):
            time.sleep(50e-3)
            self.ftdi.ftdi_fn.ftdi_usb_purge_rx_buffer()
            self.ftdi.read(16*4096)
        # start RX thread
        self.event_stop = threading.Event()
        self.thread_read = threading.Thread(target=self.read, daemon=True, name='fastreadout')
        self.thread_read.start()

    def close(self) -> None:
        self.event_stop.set()
        self.thread_read.join()
        self.ftdi.ftdi_fn.ftdi_set_bitmode(0xff, 0x00)
        self.ftdi.close()

    def read(self) -> None:
        buffer = bytearray()
        n_tot = 0
        t_start = time.perf_counter()
        while not self.event_stop.is_set():
            data_new = self.ftdi.read(3968 * 8192) # see AN232B-03 Optimising D2XX Data Throughput
            if not isinstance(data_new, bytes) or not data_new:
                time.sleep(0.001)
                continue
            n_tot += len(data_new)
            buffer.extend(data_new)
            # check if there are complete packets in the buffer
            if not b'\x00' in data_new:
                continue
            index = buffer.rindex(b'\x00')
            packets = buffer[:index].split(b'\x00')
            del buffer[:index+1]
            first_packet = True
            for packet in packets:
                try:
                    response = self._response_queue.get(False)
                    response.data = cobs.decode(packet)
                    response.event.set()
                    first_packet = False
                except queue.Empty:
                    if not first_packet:
                        print('fastro: received unexpected response', packet)
        t_end = time.perf_counter()
        t_diff = t_end - t_start
        mb_tot = n_tot / 1024**2
        print(f'fastreadout {mb_tot:0.1f}MB, {t_diff:0.1f}s, {mb_tot/t_diff:0.2f} MB/s')

    def expect_response(self) -> Response:
        response = Response()
        self._response_queue.put(response)
        return response
