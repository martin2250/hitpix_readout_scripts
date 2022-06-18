dac_default_hitpix1 = {
    'blres':  63, # 10
    'vn1':    30, # 20
    'vnfb':   63, # 10 streut stark -> amplitude
    'vnfoll': 1, # 8  --> oszillationen?
    'vndell': 8, # nicht verwendet?
    'vn2':    0,
    'vnbias': 4, # 0
    'vpload': 3, # 8 -> schneller, skaliert mit vn1
    'vncomp': 1, # 8
    'vpfoll': 1, # 16 !!
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
    'vth': 170,
}

# blres ~ vnfb --> entladung / fallende Flanke
# vpload -> ansteigende Flanke
settings_default = {
    'threshold': 1.2,
    'baseline': 1.1,
    'vssa': 1.25, # 1.85 für HP2 -> ampout messung wiederholen
    'vddd': 1.95,
    'vdda': 1.85,
    'frequency': 25.0,
    'pulse_ns': 1500.0,
}

setups = [
    'hitpix1',
    'hitpix2-1x1',
    'hitpix2-1x2-first',
    'hitpix2-1x2-last',
]

setup_dac_defaults = {
    'hitpix1': dac_default_hitpix1,
    'hitpix2-1x1': dac_default_hitpix2,
    'hitpix2-1x2-first': dac_default_hitpix2,
    'hitpix2-1x2-last': dac_default_hitpix2,
}
