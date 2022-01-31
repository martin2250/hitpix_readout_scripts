#!/usr/bin/env python
import threading
import numpy as np
import matplotlib.pyplot as plt
import gzip, base64

num_events_per_ctr = np.zeros(256)
other_pixels_values = np.zeros(256)
other_pixels_values_per_ctr = np.zeros((256, 256))
num_events_per_bit = np.zeros(8)
num_events_per_nr_bits_set = np.zeros(9)
num_events_per_bit_set = np.zeros(8)
num_events_per_nr_bits_cleared = np.zeros(9)
num_events_per_nr_bits_toggled = np.zeros(9)

events_total = 0
events_discarded = 0

filename = '../data/2022-01-30/hitpix2-1-crosstalk/log2.txt'

with open(filename, 'rb') as f:
    for line in f:
        line = line.strip()

        if line.startswith(b'#'):
            continue

        events_total += 1

        *info, data = line.split(b' ')
        test_row, test_col, i_frame_abs = map(int, info)

        # absolute counter values before and after injection
        frames = np.frombuffer(
            gzip.decompress(
                base64.b64decode(data)
            ),
            dtype=np.uint8,
        ).reshape(2, 48, 48)

        frames_diff = np.diff(frames, axis=0)

        # raw counter value before
        num_events_per_ctr[frames[0, test_row, test_col]] += 1

        # which bits were set before the event?
        for i_bit in range(8):
            if (frames[0, test_row, test_col] >> i_bit) & 0x01:
                num_events_per_bit[i_bit] += 1

        # which bits were set to 1/0/toggled during the injection?
        before, after = frames[:, test_row, test_col]
        bits_set = int(after & (~before))
        bits_cleared = int(before & (~after))
        bits_toggled = int(before ^ after)
        num_events_per_nr_bits_set[bits_set.bit_count()] += 1
        num_events_per_nr_bits_cleared[bits_cleared.bit_count()] += 1
        num_events_per_nr_bits_toggled[bits_toggled.bit_count()] += 1

        for i_bit in range(8):
            if (bits_set >> i_bit) & 0x01:
                num_events_per_bit_set[i_bit] += 1

        # what values did the other pixels in the column have?
        other_pixels = np.copy(frames)
        # exclude triggered pixel
        other_pixels[:, test_row, test_col] = 0
        # which pixels changed?
        other_pixels_diff = np.diff(other_pixels, axis=0)

        # discard events where a pixel was hit and had counter == 0 before injection
        pixel_zero_hit = (frames_diff > 0) & (frames[0] == 0)
        if np.any(pixel_zero_hit):
            events_discarded += 1
            continue

        # histogram over changed pixel values
        other_pixels_hist = np.bincount(
            other_pixels[:1][other_pixels_diff > 0],
            minlength=len(other_pixels_values),
        )
        other_pixels_values += other_pixels_hist
        other_pixels_values_per_ctr[frames[0, test_row, test_col]] += other_pixels_hist

print(f'{events_total=} {events_discarded=}')

if False:
    plt.bar(np.arange(8), num_events_per_bit_set / np.sum(num_events_per_ctr), width=1)
    plt.xlabel('which bit set to 1 during injection')

if False:
    plt.bar(np.arange(9), num_events_per_nr_bits_set / np.sum(num_events_per_ctr), width=1)
    plt.xlabel('number of bits set to 1 during injection')

if False:
    plt.bar(np.arange(9), num_events_per_nr_bits_cleared / np.sum(num_events_per_ctr), width=1)
    plt.xlabel('number of bits set to 0 during injection')

if False:
    plt.bar(np.arange(9), num_events_per_nr_bits_toggled / np.sum(num_events_per_ctr), width=1)
    plt.xlabel('number of bits toggled during injection')

if False:
    plt.bar(np.arange(8), num_events_per_bit / np.sum(num_events_per_ctr), width=1)
    plt.xlabel('number of bits set before injection')

if True:
    plt.bar(np.arange(256), num_events_per_ctr / np.sum(num_events_per_ctr), width=1)
    plt.xlabel('counter value before injection')
    plt.semilogy()

if False:
    plt.bar(np.arange(256), other_pixels_values, width=1)
    plt.semilogy()
    plt.xlabel('counter values of noise hits before injection')

if False:
    import matplotlib.colors
    plt.imshow(other_pixels_values_per_ctr.T, norm=matplotlib.colors.LogNorm())
    plt.xlabel('counter values of PuT before injection')
    plt.ylabel('counter values of noise hits before injection')
        

plt.axhline(0.5)
plt.ylabel('fraction X of events with crosstalk')
plt.show()