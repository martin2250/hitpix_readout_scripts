#!/usr/bin/env python
import queue
import sys
from pathlib import Path

if True:  # do not reorder with autopep8 or sortimports
    sys.path.insert(1, str(Path(__file__).parents[1]))

import time
import unittest

import hitpix
import hitpix.defaults
import util.configuration
import util.gridscan
import util.voltage_channel
from hitpix.readout import HitPixReadout
from readout.fast_readout import FastReadout
from readout.instructions import *
from readout.sm_prog import *

############################################################################

cfg_setup = 'hitpix1'

############################################################################

class TestReadoutSmPackets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config_readout = util.configuration.load_config()
        serial_port_name, board = config_readout.find_board()

        cls.fastreadout = FastReadout(board.fastreadout_serial_number)
        time.sleep(0.05)
        cls.addClassCleanup(cls.fastreadout.close)

        cls.ro = HitPixReadout(serial_port_name, hitpix.setups[cfg_setup])
        cls.ro.initialize()
        cls.addClassCleanup(cls.ro.close)
    
    def test_finite(self):
        self.ro.sm_write([
            GetTime(),
            Sleep(100),
        ] * 16 + [
            Finish(),
        ])
        # initial run
        re = self.fastreadout.expect_response()
        self.ro.sm_start(7) # use different number of executions here
        self.ro.wait_sm_idle()
        self.assertTrue(re.event.wait(1.0))
        # test different packets
        for num_packets in [1, 5, 25]:
            for num_exec in [1, 20, 127]:
                resp = [
                    self.fastreadout.expect_response()
                    for _ in range(num_packets)
                ]
                self.ro.sm_start(num_exec, packets=num_packets)
                self.ro.wait_sm_idle()
                for i, re in enumerate(resp):
                    self.assertTrue(re.event.wait(2.0))
                    self.assertIsNotNone(re.data)
                    assert re.data is not None
                    self.assertEqual(len(re.data), 16 * 4 * num_exec, f'{i=} {num_packets=}, {num_exec=}')
    
    def test_infinite(self):
        ftdi_speed = 4 # Mbyte/s
        cycles_per_word = int(4 * self.ro.frequency_mhz / ftdi_speed)
        # sm prog: write 16 32bit words
        self.ro.sm_write([
            GetTime(),
            Sleep(cycles_per_word),
        ] * 16 + [
            Finish(),
        ])
        # test 1 or 2 executions (there were problems with juts one)
        for num_exec in [1, 2]:
            # claim responses
            responses = queue.Queue()
            self.fastreadout.orphan_response_queue = responses
            # start infinite number of packets
            self.ro.sm_start(num_exec, packets=0)
            # wait a short time
            time.sleep(0.1)
            # check if it is still running
            self.assertTrue(self.ro.get_sm_status().active)
            # soft abort
            self.ro.sm_soft_abort()
            # wait for sm to go idle, properly abort when not idle
            try:
                self.ro.wait_sm_idle(1.0)
            except TimeoutError as ex:
                self.ro.sm_abort()
                self.ro.wait_sm_idle(1.0)
                raise ex
            # check responses
            num = 0
            while True:
                try:
                    re = responses.get(timeout=0.2)
                except queue.Empty:
                    break
                num += 1
                self.assertTrue(isinstance(re, bytes))
                self.assertEqual(len(re), 16 * 4 * num_exec)
            # reset fastreadout
            self.fastreadout.orphan_response_queue = None
            self.assertGreater(num, 100)


############################################################################


if __name__ == '__main__':
    unittest.main()
