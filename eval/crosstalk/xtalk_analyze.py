#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt


num_events = np.zeros(256)
num_hits_col = np.zeros(256)
num_hits_sens = np.zeros(256)

filename = '../data/2022-01-30/hitpix2-1-crosstalk/log.txt'

with open(filename) as f:
    for line in f:
        line = line.strip()
        if line.startswith('otherpixel') or line.startswith('running') or line.startswith('dac_cfg'):
            continue
        i_frame, testpixel, testpixel_abs, otherpixels, otherpixels_col = map(int, line.split())
        if testpixel == 1:
            num_events[testpixel_abs] += 1
            num_hits_col[testpixel_abs] += otherpixels_col
            num_hits_sens[testpixel_abs] += otherpixels
        elif testpixel != 0:
            print(testpixel, otherpixels, otherpixels_col)


# num_events_per_bit = np.zeros(8)
# for i_bit in range(8):
#     for i, v in enumerate(num_events):
#         if ((i >> i_bit) & 0x01) == 0x01:
#             num_events_per_bit[i_bit] += v
    
# plt.bar(np.arange(8), num_events_per_bit / np.sum(num_events), width=1)
        

plt.bar(np.arange(256), num_events, width=1)
plt.semilogy()
plt.axhline(0.5)
plt.ylabel('fraction X of events with crosstalk had bit N set')
plt.xlabel('bit [n] set in PuT counter beforehand')
plt.show()