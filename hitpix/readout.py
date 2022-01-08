from enum import IntEnum
from readout.readout import Readout
from readout.dac_card import DacCard
from . import HitPixSetup


class HitPix1VoltageCards(IntEnum):
    threshold = 1
    baseline  = 4

class HitPix1Pins(IntEnum):
    ro_ldconfig = 0
    ro_psel     = 1
    ro_penable  = 2
    ro_ldcnt    = 3
    ro_rescnt   = 4
    ro_frame    = 5
    dac_ld      = 6
    dac_inv_ck  = 30
    ro_inv_ck   = 31

class HitPixReadout(Readout):
    def __init__(self, serial_name: str, setup: HitPixSetup, timeout: float = 0.5) -> None:
        super().__init__(serial_name, timeout=timeout)
        self.setup = setup
        self.dac_cards = {}

        # voltage cards
        for i_slot, _ in [setup.vc_baseline, setup.vc_baseline]:
            if i_slot not in self.dac_cards:
                self.dac_cards[i_slot] = DacCard(i_slot, 8, 3.3, 1.8, self)

        # injection card(s)
        for i_slot, _ in [setup.vc_injection]:
            if i_slot not in self.dac_cards:
                self.dac_cards[i_slot] = DacCard(i_slot, 2, 3.3, 1.8, self)

        # set inverted pins
        self.write_register(self.ADDR_SM_INVERT_PINS, setup.invert_pins)

        self.frequency_mhz = 200

    def set_threshold_voltage(self, voltage: float) -> None:
        i_slot, i_chan = self.setup.vc_threshold
        self.dac_cards[i_slot].set_voltage(i_chan, voltage)

    def set_baseline_voltage(self, voltage: float) -> None:
        i_slot, i_chan = self.setup.vc_baseline
        self.dac_cards[i_slot].set_voltage(i_chan, voltage)

    def set_injection_voltage(self, voltage: float) -> None:
        i_slot, i_chan = self.setup.vc_injection
        self.dac_cards[i_slot].set_voltage(i_chan, voltage)

    def initialize(self) -> None:
        super().initialize()
        version = self.get_version()
        if self.setup.chip.version_number != version.chip:
            raise RuntimeError(
                f'FPGA bitstream chip version ({version.chip}) does not match setup ({self.setup.chip.version_number})')
        if self.setup.version_number != version.adapter:
            raise RuntimeError(
                f'FPGA bitstream adapter version ({version.adapter}) does not match setup ({self.setup.version_number})')
        if version.readout != 8:
            raise RuntimeError(f'unsupported readout version {version.readout}')
