import queue
import threading
import time

import cobs
import pylibftdi

from . import Response


class FastReadout:
    def __init__(self, serial_number: str) -> None:
        self._response_queue: queue.Queue[Response] = queue.Queue()
        # open FTDI
        FT_FLOW_RTS_CTS = 0x0100
        self.ftdi = pylibftdi.Device(serial_number)
        self.ftdi.ftdi_fn.ftdi_set_bitmode(0xff, 0x00)
        time.sleep(10e-3)
        self.ftdi.ftdi_fn.ftdi_set_bitmode(0xff, 0x40)
        self.ftdi.ftdi_fn.ftdi_setflowctrl(FT_FLOW_RTS_CTS, 0, 0)

        # start RX thread
        threading.Thread(target=self.read, daemon=True,
                         name='fastreadout').start()

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
