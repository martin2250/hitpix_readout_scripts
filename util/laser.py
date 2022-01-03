import enum

import serial


class NktPiLaser:
    def __init__(self, port_name: str) -> None:
        self.port = serial.Serial(port_name, 19200)

    ############################################################################

    def query(self, command: str) -> str:
        self.port.write(command.encode())
        self.port.flushOutput()
        res = self.port.readall()
        return res.decode()

    def write(self, command: str):
        '''query and expect "done" as reply'''
        response = self.query(command)
        if response != 'done':
            raise RuntimeError(f'invalid response {command=} {response=}')

    ############################################################################

    @property
    def version(self) -> str:
        return self.query('version?')

    ############################################################################

    @property
    def state(self) -> bool:
        resp = self.query('ld?')
        print(resp)
        return 'on' in resp

    @state.setter
    def state(self, state_new: bool) -> None:
        self.write(f'ld={int(state_new)}')

    ############################################################################

    @property
    def tune(self) -> int:
        resp = self.query('tune?')
        print(resp)
        return int(resp)

    @tune.setter
    def tune(self, tune_new: int) -> None:
        self.write(f'tune={tune_new}')

    ############################################################################

    @property
    def trigger_frequency(self) -> int:
        resp = self.query('f?')
        print(resp)
        return int(resp)

    @trigger_frequency.setter
    def trigger_frequency(self, freq_new: int) -> None:
        self.write(f'f={freq_new}')

    ############################################################################

    class TriggerEdge(enum.IntEnum):
        FALLING = 0
        RISING = 1

    @property
    def trigger_edge(self) -> TriggerEdge:
        resp = self.query('te?')
        print(resp)
        if 'rising' in resp:
            return NktPiLaser.TriggerEdge.RISING
        else:
            return NktPiLaser.TriggerEdge.FALLING

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
        resp = self.query('ts?')
        print(resp)
        if 'internal' in resp:
            return NktPiLaser.TriggerSource.INTERNAL
        elif 'ext. adjustable' in resp:
            return NktPiLaser.TriggerSource.EXTERNAL_ADJ
        elif 'ext. TTL' in resp:
            return NktPiLaser.TriggerSource.EXTERNAL_TTL
        else:
            raise RuntimeError(f'unknown trigger source {resp=}')

    @trigger_source.setter
    def trigger_source(self, source_new: TriggerSource) -> None:
        self.write(f'ts={int(source_new)}')

    ############################################################################

    @property
    def trigger_level(self) -> float:
        resp = self.query('tl?')
        *_, thresh, unit = resp.split()
        if unit == 'mV':
            return float(thresh) / 1000
        elif unit == 'V':
            return float(thresh)
        else:
            raise RuntimeError(f'unknown unit {resp=}')

    @trigger_level.setter
    def trigger_level(self, value: float) -> None:
        assert value > -4.8 and value < 4.8
        self.write(f'tl={int(value*1000)}')
