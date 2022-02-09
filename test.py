import numpy as np
import matplotlib.pyplot as plt

from util.xilinx.pll7series import get_register_values, optimize_vco_and_divider

freq_set, freq_is = np.array([
    (55, 40),
    (38.02, 29.17),
    (65.625, 50),
    (100, 75),
    (21, 17.85),
    (15, 10),
    (46.09, 37.5),
]).T

# print(freq_is / freq_set)

# updating readout frequency div_fb=7.375 2*div_out=8 bitrate=46.09375 Mbit/s
# --> actual VCO frequency: 600MHz

# updating readout frequency div_fb=7.875 2*div_out=6 bitrate=65.625 Mbit/s
# --> actual VCO frequency: 600MHz

# request 78
# updating readout frequency div_fb=6.25 2*div_out=4 bitrate=78.125 Mbit/s
# --> actual VCO frequency: 400MHz, bitrate: 50Mbit

frequency_mhz = 78
div_fb, div_out, freq_gen = optimize_vco_and_divider(100.0, 4 * frequency_mhz)
print(f'updating readout frequency {div_fb=} {2*div_out=} bitrate={freq_gen/4} Mbit/s')
regs = get_register_values(div_fb, div_out, 'optimized')