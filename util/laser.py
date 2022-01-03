import enum

import serial


class NktPiLaser:
    def __init__(self, port_name: str) -> None:
        self.port = serial.Serial(port_name, 19200)
        self.port.timeout = 0.5

    ############################################################################

    def query(self, command: str, expect_lines: int = 1) -> list[str]:
        command = command + '\r\n'
        self.port.write(command.encode())
        self.port.flushOutput()
        response = [
            self.port.readline().decode().strip()
            for _ in range(expect_lines)
        ]
        return response

    def write(self, command: str):
        '''query and expect "done" as reply'''
        response, = self.query(command, 1)
        if response != 'done':
            raise RuntimeError(f'invalid response {command=} {response=}')

    def read(self, command: str) -> tuple[str, str]:
        response, = self.query(command)
        prefix, _, value = response.partition(':')
        return prefix.strip(), value.strip()

    ############################################################################

    def get_version(self) -> list[str]:
        return self.query('version?', 8)

    ############################################################################

    @property
    def state(self) -> bool:
        resp, = self.query('ld?')
        # pulsed laser emission: on
        # pulsed laser emission: off
        return resp.endswith('on')

    @state.setter
    def state(self, state_new: bool) -> None:
        self.write(f'ld={int(state_new)}')

    ############################################################################

    @property
    def tune(self) -> float:
        prefix, value = self.read('tune?')
        # tune value:\t\t     20.00 %
        assert prefix == 'tune value', prefix
        factor = 1
        if value.endswith('%'):
            value = value[:-1].strip()
            factor = 1 / 100
        return factor * float(value)

    @tune.setter
    def tune(self, tune_new: float) -> None:
        self.write(f'tune={int(1000*tune_new):d}')

    ############################################################################

    @property
    def trigger_frequency(self) -> int:
        prefix, value = self.read('f?')
        # int. frequency:\t      1000 Hz
        assert prefix == 'int. frequency', prefix
        assert value.endswith('Hz'), value
        return int(value[:-2].strip())

    @trigger_frequency.setter
    def trigger_frequency(self, freq_new: int) -> None:
        self.write(f'f={freq_new}')

    ############################################################################

    class TriggerEdge(enum.IntEnum):
        FALLING = 0
        RISING = 1

    @property
    def trigger_edge(self) -> TriggerEdge:
        prefix, value = self.read('te?')
        # trigger edge:\trising
        assert prefix == 'trigger edge', prefix
        if value == 'rising':
            return NktPiLaser.TriggerEdge.RISING
        elif value == 'falling':
            return NktPiLaser.TriggerEdge.FALLING
        raise RuntimeError(f'invalid trigger edge {value=}')

    @trigger_edge.setter
    def trigger_edge(self, edge_new: TriggerEdge) -> None:
        self.write(f'te={int(edge_new)}')

    ############################################################################

    class TriggerSource(enum.IntEnum):
        INTERNAL = 0
        EXTERNAL_ADJ = 1
        EXTERNAL_TTL = 2

    @property
    def trigger_source(self) -> TriggerSource:
        prefix, value = self.read('ts?')
        # trigger source:\tinternal
        assert prefix == 'trigger source', prefix
        if value == 'internal':
            return NktPiLaser.TriggerSource.INTERNAL
        elif value == 'ext. adjustable':
            return NktPiLaser.TriggerSource.EXTERNAL_ADJ
        elif value == 'ext. TTL':
            return NktPiLaser.TriggerSource.EXTERNAL_TTL
        else:
            raise RuntimeError(f'unknown trigger source {value=}')

    @trigger_source.setter
    def trigger_source(self, source_new: TriggerSource) -> None:
        self.write(f'ts={int(source_new)}')

    ############################################################################

    @property
    def trigger_level(self) -> float:
        prefix, value = self.read('tl?')
        # trigger level:\t     +0.80 V
        assert prefix == 'trigger level', prefix
        thresh, unit = value.split()
        if unit == 'mV':
            return float(thresh) / 1000
        elif unit == 'V':
            return float(thresh)
        else:
            raise RuntimeError(f'unknown unit {unit=}')

    @trigger_level.setter
    def trigger_level(self, value: float) -> None:
        assert value > -4.8 and value < 4.8
        self.write(f'tl={int(value*1000)}')
