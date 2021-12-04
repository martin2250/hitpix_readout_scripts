import configparser
from dataclasses import dataclass
from typing import Optional
import os.path
import pathlib

@dataclass
class ReadoutBoardConfig:
    fastreadout_serial_number: str

@dataclass
class HitpixReadoutConfig:
    serial_baud: int
    boards: dict[str, ReadoutBoardConfig] # key = serial port name
    fpga_flash_command: Optional[str] = None

    def find_board(self) -> tuple[str, ReadoutBoardConfig]:
        for portname, config in self.boards.items():
            portpath = pathlib.Path(portname)
            if portpath.exists():
                return portname, config
        raise RuntimeError('no matching board found')

def load_config() -> HitpixReadoutConfig:
    # set defaults
    config = configparser.ConfigParser()
    config['DEFAULT']['serial_baud'] = str(3_000_000)
    # check if config file exists
    config_path = pathlib.Path('~/.config/hitpix_readout.ini').expanduser()
    if not config_path.exists():
        print('hitpix configuration file not found.')
        res = input(f'generate configuration template {config_path}? [y/N]: ')
        if res.lower() != 'y':
            exit()
        config['/dev/serial/by-id/example']['fastreadout_serial'] = 'EXAMPLE'
        with config_path.open('w') as configfile:
            config.write(configfile)
        print('example configuration written')
        exit()
    # read config file
    config.read(os.path.expanduser(config_path))
    # parse DEFAULT
    fpga_flash_command = config['DEFAULT'].get('fpga_flash_command')
    serial_baud = int(config['DEFAULT']['serial_baud'])
    # parse boards
    boards = {}
    for section in config.sections():
        if section == 'DEFAULT':
            continue
        boards[section] = ReadoutBoardConfig(config[section]['fastreadout_serial'])

    return HitpixReadoutConfig(
        serial_baud=serial_baud,
        boards=boards,
        fpga_flash_command=fpga_flash_command,
    )
