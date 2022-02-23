#!/usr/bin/env python
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import sys

################################################################################
# load file

with open(sys.argv[1]) as f_in:
    assert next(f_in).startswith('# latency scan')
    for line in f_in:
        if line.startswith('# latencies'):
            break
    else:
        assert False
    _, *lat_str = line.strip().split('\t')
    latencies = list(int(s.strip()) for s in lat_str)
    frequencies, *errors = np.loadtxt(f_in, unpack=True)

frequencies = np.array(frequencies)
errors = np.array(errors).T.copy()

################################################################################
# find center of working latencies for each frequency

res_freq = []
res_latency = []

for freq, err in zip(frequencies, errors):
    max_run_center = -1
    max_run_length = -1
    # find largest run of err == 0
    pos = 0
    group_last_len = 0
    for ok, group in itertools.groupby(err == 0):
        # make copy, so iter can be reused
        group = list(group)
        # keep track of position
        pos += group_last_len
        group_last_len = len(group)
        # discard False groups
        if not ok:
            continue
        if len(group) < max_run_length:
            continue
        # found next candidate
        max_run_length = len(group)
        max_run_center = pos + len(group) / 2
    if max_run_length < 0:
        continue
    res_freq.append(freq)
    res_latency.append(max_run_center)

################################################################################
# fit polynomial

polyvals = np.polyfit(res_freq, res_latency, 3)
polyvals_str = ', '.join(f'{v:0.3e}' for v in polyvals)
print(f'polyvals = [{polyvals_str}]')

################################################################################
# plot results

X_plot = np.linspace(min(res_freq), max(res_freq), 100)
Y_plot = np.poly1d(polyvals)(X_plot)

plt.plot(res_freq, res_latency, '+')
plt.plot(X_plot, Y_plot)
plt.savefig('test.png')
