#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import sys

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

part = slice(4, None)
latencies = np.array(latencies[part])
errors = np.array(errors[part])

max_error = np.max(errors) * 1.3


errors_best = np.min(errors, axis=0)
max_freq = frequencies[np.argmin(errors_best == 0)]
print(f'{max_freq=}')


cmap = matplotlib.cm.get_cmap('YlOrRd').copy() # type: ignore
cmap.set_bad('green')

plt.pcolormesh(
    frequencies,
    latencies,
    np.where(errors == 0, np.nan, errors),
    cmap=cmap,
)
plt.suptitle('Error Rate vs Data Rate')
plt.xlabel('Readout Bitrate (Mbit/s)')
plt.ylabel('Sampling Latency (quarter bits)')
plt.tight_layout()
plt.savefig('hitpix1.png')
# plt.show()

# for latency, error in zip(latencies[part], errors[part]):
#     plt.plot(frequencies, error + max_error*latency, label=f'{latency}')

# plt.legend()
# plt.show()