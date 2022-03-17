import configparser
from dataclasses import dataclass
from typing import Optional
import os.path
import pathlib

@dataclass
class ReadoutBoardConfig:
    fastreadout_serial_number: str
    default_hv_driver: str
    default_vddd_driver: str
    default_vdda_driver: str
    default_vssa_driver: str
    laser_port: str
    motion_port: str

@dataclass
class HitpixReadoutConfig:
    serial_baud: int
    boards: dict[str, ReadoutBoardConfig] # key = serial port name
    fpga_flash_command: Optional[str] = None
    telegram: Optional[tuple[str, str]] = None # token and chatid

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
    default = config['DEFAULT']
    # parse DEFAULT
    fpga_flash_command = default.get('fpga_flash_command')
    telegram = None
    # TODO: move telegram to own section
    if 'telegram_token' in config:
        telegram = default['telegram_token'], default['telegram_chatid']
    serial_baud = int(default['serial_baud'])
    # parse boards
    boards = {}
    for section in config.sections():
        if section == 'DEFAULT':
            continue
        conf_section = config[section]
        fastreadout_serial = conf_section.get('fastreadout_serial')
        default_hv_driver = conf_section.get('default_hv_driver', 'manual')
        default_vddd_driver = conf_section.get('default_vddd_driver', 'manual')
        default_vdda_driver = conf_section.get('default_vdda_driver', 'manual')
        default_vssa_driver = conf_section.get('default_vssa_driver', 'manual')
        motion_port = conf_section.get('motion_port', '')
        laser_port = conf_section.get('laser_port', '')
        boards[section] = ReadoutBoardConfig(
            fastreadout_serial,
            default_hv_driver,
            default_vddd_driver=default_vddd_driver,
            default_vdda_driver=default_vdda_driver,
            default_vssa_driver=default_vssa_driver,
            motion_port=motion_port,
            laser_port=laser_port,
        )

    return HitpixReadoutConfig(
        serial_baud=serial_baud,
        boards=boards,
        fpga_flash_command=fpga_flash_command,
        telegram=telegram,
    )
