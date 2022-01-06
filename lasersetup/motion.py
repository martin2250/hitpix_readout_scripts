import enum
import logging
import time

import serial

logger = logging.getLogger(__name__)


class PiSerial:
    def __init__(self, port: serial.Serial) -> None:
        self.port = port
        self.port.write('\n\n'.encode())
        self.port.flushOutput()

    def write(self, id: int, cmd: str) -> None:
        cmd_enc = f'{id} 0 {cmd}\n'.encode()
        logger.debug(f'write {cmd_enc}')
        if self.port.write(cmd_enc) != len(cmd_enc):
            raise Exception('could not write to port')
        self.port.flushOutput()

    def query(self, id: int, cmd: str) -> str:
        '''does not return address stuff'''
        resp_prefix = f'0 {id} '.encode()
        self.write(id, cmd)
        resp = self.port.readline().strip()
        logger.debug(f'receive {resp}')
        if not resp.startswith(resp_prefix):
            raise Exception(
                f'response does not start with prefix ({resp_prefix}): "{resp}"')
        return resp[len(resp_prefix):].decode()


class HomePos(enum.Enum):
    NEGATIVE = 1
    MIDDLE = 2
    POSITIVE = 3


class MotorAxis:
    def __init__(self, port_id: int, port: PiSerial, motion_range: tuple[float, float], backlash: float, home_pos: HomePos) -> None:
        self.port_id = port_id
        self.port = port
        self.motion_range = motion_range
        self.backlash = backlash  # positive: approach points from negative coordinates
        self.position = float('nan')
        self.home_pos = home_pos

    def query_on_target(self) -> bool:
        rsp_ont = self.port.query(self.port_id, 'ONT?')
        responses = {'1=0': False, '1=1': True}
        if rsp_ont not in responses:
            raise Exception(f'invalid response "{rsp_ont}"')
        return responses[rsp_ont]

    def query_position(self) -> float:
        rsp_pos = self.port.query(self.port_id, 'POS?')
        rsp_start = '1='
        if not rsp_pos.startswith(rsp_start):
            raise Exception(f'invalid response "{rsp_pos}"')
        return float(rsp_pos[len(rsp_start):])

    def query_homed(self) -> bool:
        rsp_ref = self.port.query(self.port_id, 'FRF? 1')
        responses = {'1=0': False, '1=1': True}
        if rsp_ref not in responses:
            raise Exception(f'invalid response "{rsp_ref}"')
        return responses[rsp_ref]

    def query_motor_state(self) -> bool:
        rsp_ref = self.port.query(self.port_id, 'SVO? 1')
        responses = {'1=0': False, '1=1': True}
        if rsp_ref not in responses:
            raise Exception(f'invalid response "{rsp_ref}"')
        return responses[rsp_ref]

    def query_error(self) -> int:
        return int(self.port.query(self.port_id, 'ERR?'))

    def wait_on_target(self):
        while not self.query_on_target():
            time.sleep(0.01)

    def set_motor_state(self, active: bool):
        active_str = '1' if active else '0'
        self.port.write(self.port_id, f'SVO 1 {active_str}')

    def move_to_raw(self, pos_new: float) -> None:
        self.position = pos_new
        self.port.write(self.port_id, f'MOV 1 {pos_new:0.3f}')

    def move_to(self, pos_new: float, wait: bool = False) -> None:
        pos_diff = pos_new - self.position
        # check for zero length move
        if pos_diff == 0:
            return
        # optional backlash compensation
        if (pos_diff > 0) != (self.backlash > 0):
            self.move_to_raw(pos_new - self.backlash)
            self.wait_on_target()
        self.move_to_raw(pos_new)
        if wait:
            self.wait_on_target()

    def move_home(self) -> None:
        cmd = {
            HomePos.NEGATIVE: "FNL",
            HomePos.MIDDLE: "FRF",
            HomePos.POSITIVE: "FPL",
        }[self.home_pos]
        self.port.write(self.port_id, f'{cmd} 1')

    def initialize(self) -> None:
        if not self.query_motor_state():
            self.set_motor_state(True)
        if not self.query_homed():
            self.move_home()
            self.wait_on_target()
        self.position = self.query_position()


class MotorStage:
    def __init__(self, axis_x: MotorAxis, axis_y: MotorAxis, axis_z: MotorAxis):
        self.axis_x = axis_x
        self.axis_y = axis_y
        self.axis_z = axis_z
        self.axes = [axis_x, axis_y, axis_z]

    def query_on_target(self):
        for axis in self.axes:
            if not axis.query_on_target():
                return False
        return True

    def wait_on_target(self):
        for axis in self.axes:
            axis.wait_on_target()

    def initialize(self):
        wait_on_target = False
        for axis in self.axes:
            if not axis.query_motor_state():
                axis.set_motor_state(True)
            if not axis.query_homed():
                axis.move_home()
                wait_on_target = True
        if wait_on_target:
            self.wait_on_target()
        for axis in self.axes:
            axis.position = axis.query_position()

    def move_to_raw(self, x: float, y: float, z: float, wait: bool = False):
        self.axis_x.move_to_raw(x)
        self.axis_y.move_to_raw(y)
        self.axis_z.move_to_raw(z)
        if wait:
            self.wait_on_target()

    def move_to(self, x: float, y: float, z: float, wait: bool = False):
        pos_new = [x, y, z]
        # which axes need to be moved?
        axis_needs_to_move = [
            ax.position != pn
            for ax, pn in zip(self.axes, pos_new)
        ]
        axis_moving = [False for _ in pos_new]
        # perform backlash compensation
        for i, axis in enumerate(self.axes):
            if not axis_needs_to_move[i]:
                continue
            pos_diff = pos_new[i] - axis.position
            # approaching from wrong side?
            if (pos_diff > 0) != (axis.backlash > 0):
                # overshoot target a bit
                axis.move_to_raw(pos_new[i] - axis.backlash)
                axis_moving[i] = True
            else:
                # move to target directly
                axis.move_to_raw(pos_new[i])
                axis_needs_to_move[i] = False
                axis_moving[i] = True
        # wait on axes that had backlash compensated
        for i, axis in enumerate(self.axes):
            if not axis_needs_to_move[i]:
                continue
            axis.wait_on_target()
        # move to actual position
        for i, axis in enumerate(self.axes):
            if not axis_needs_to_move[i]:
                continue
            axis.move_to_raw(pos_new[i])
        # wait on target
        if not wait:
            return
        for i, axis in enumerate(self.axes):
            if not axis_moving:
                continue
            axis.wait_on_target()

def load_motion(port: PiSerial | serial.Serial | str) -> MotorStage:
    '''stage must be initialized afterwards!'''
    if isinstance(port, str):
        port = serial.Serial(port, baudrate=9600, timeout=1)
    if isinstance(port, serial.Serial):
        port = PiSerial(port)
    if not isinstance(port, PiSerial):
        raise ValueError('invalid type for port')
    axis_x = MotorAxis(3, port, (0., 100.), 0.1, HomePos.MIDDLE)
    axis_y = MotorAxis(2, port, (0., 25.), 0.1, HomePos.NEGATIVE)
    axis_z = MotorAxis(1, port, (0., 20.), -0.1, HomePos.NEGATIVE) # negative backlash -> approach positions by lifting up
    stage = MotorStage(axis_x, axis_y, axis_z)
    return stage
