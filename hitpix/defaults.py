dac_default_hitpix1 = {
    'blres':  63,
    'vn1':    30,
    'vnfb':   63,
    'vnfoll': 1,
    'vndell': 8,
    'vn2':    0,
    'vnbias': 4,
    'vpload': 3,
    'vncomp': 1,
    'vpfoll': 1,
}

dac_default_hitpix2 = {
    'enable_output_cmos': True,
    'enable_output_diff': True,
    'enable_bandgap': False,
    'unlock': 0b101,
    'iblres': 63,
    'vn1': 30,
    'infb': 63,
    'vnfoll': 1,
    'vndel': 8,
    'vn2': 0,
    'infb2': 4,
    'ipload2': 3,
    'vncomp': 1,
    'ipfoll': 1,
    'vth': 127,
}

voltages_default = {
    'threshold': 1.2,
    'baseline': 1.1,
    'vssa': 1.25,
    'vdd': 1.85,
}

setups = [
    'hitpix1',
    'hitpix2-1x1',
    'hitpix2-1x5',
]