#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import gzip, base64

# https://github.com/google/jax/blob/6c8fc1b031275c85b02cb819c6caa5afa002fa1d/jax/lax_reference.py#L121-L150
def population_count(x):
    assert np.issubdtype(x.dtype, np.integer)
    dtype = x.dtype
    iinfo = np.iinfo(x.dtype)
    if np.iinfo(x.dtype).bits < 32:
        assert iinfo.kind in ('i', 'u')
        x = x.astype(np.uint32 if iinfo.kind == 'u' else np.int32)
    if iinfo.kind == 'i':
        x = x.view(f"uint{np.iinfo(x.dtype).bits}")
    assert x.dtype in (np.uint32, np.uint64)
    m = [
        np.uint64(0x5555555555555555),  # binary: 0101...
        np.uint64(0x3333333333333333),  # binary: 00110011..
        np.uint64(0x0f0f0f0f0f0f0f0f),  # binary:  4 zeros,  4 ones ...
        np.uint64(0x00ff00ff00ff00ff),  # binary:  8 zeros,  8 ones ...
        np.uint64(0x0000ffff0000ffff),  # binary: 16 zeros, 16 ones ...
        np.uint64(0x00000000ffffffff),  # binary: 32 zeros, 32 ones
    ]

    if x.dtype == np.uint32:
        m = list(map(np.uint32, m[:-1]))

    x = (x & m[0]) + ((x >>  1) & m[0])  # put count of each  2 bits into those  2 bits
    x = (x & m[1]) + ((x >>  2) & m[1])  # put count of each  4 bits into those  4 bits
    x = (x & m[2]) + ((x >>  4) & m[2])  # put count of each  8 bits into those  8 bits
    x = (x & m[3]) + ((x >>  8) & m[3])  # put count of each 16 bits into those 16 bits
    x = (x & m[4]) + ((x >> 16) & m[4])  # put count of each 32 bits into those 32 bits
    if x.dtype == np.uint64:
        x = (x & m[5]) + ((x >> 32) & m[5])  # put count of each 64 bits into those 64 bits
    return x.astype(dtype)

num_events_per_pixel_put = np.zeros((48, 48))
num_events_per_pixel_recv = np.zeros((48, 48))
num_events_per_ctr = np.zeros(256)
other_pixels_values = np.zeros(256)
other_pixels_values_per_ctr = np.zeros((256, 256))
num_events_per_bit = np.zeros(8)
num_events_per_nr_bits_set = np.zeros(9)
num_events_per_bit_set = np.zeros(8)
num_events_per_nr_bits_cleared = np.zeros(9)
num_events_per_nr_bits_toggled = np.zeros(9)

num_hits_per_adder_bits_set = np.zeros(15)
num_hits_per_adder_bits_cleared = np.zeros(15)
num_hits_per_adder_bits_toggled = np.zeros(15)

hits_above_put = 0
hits_below_put = 0
evts_above_put = 0
evts_below_put = 0
evts_both_put = 0

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
        num_events_per_pixel_put[test_row, test_col] += 1

        frames_diff_wo_put = frames_diff[0].copy()
        frames_diff_wo_put[test_row, test_col] = 0
        num_events_per_pixel_recv += frames_diff_wo_put

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


        # noise in column above PuT or below?
        above_put = np.sum(frames_diff[0,:test_row,test_col])
        below_put = np.sum(frames_diff[0,test_row+1:,test_col])
        hits_above_put += above_put
        hits_below_put += below_put
        evts_above_put += above_put > 0
        evts_below_put += below_put > 0
        evts_both_put += above_put > 0 and below_put > 0

        # check adders
        # calculate running sum of counter values
        adders = np.cumsum(frames, axis=1)
        # we want the adder inputs at each pixel
        # -> roll array by one and set first row to zero
        adders = np.roll(adders, 1, axis=1)
        adders[:,0,:] = 0
        # check how many bits were newly set/toggled during injection
        # "bit set after injection and not before injection"
        adders_set = adders[1] & (~adders[0])
        adders_cleared = adders[0] & (~adders[1])
        adders_toggled = adders[1] ^ adders[0]
        adders_set_popcnt = population_count(adders_set).astype(np.int64)
        adders_toggled_popcnt = population_count(adders_toggled).astype(np.int64)
        adders_cleared_popcnt = population_count(adders_cleared).astype(np.int64)
        # histogram this for each hit
        num_hits_per_adder_bits_set += np.bincount(
            adders_set_popcnt[frames_diff[0] > 0],
            minlength=len(num_hits_per_adder_bits_set),
        )
        num_hits_per_adder_bits_cleared += np.bincount(
            adders_cleared_popcnt[frames_diff[0] > 0],
            minlength=len(num_hits_per_adder_bits_cleared),
        )
        num_hits_per_adder_bits_toggled += np.bincount(
            adders_toggled_popcnt[frames_diff[0] > 0],
            minlength=len(num_hits_per_adder_bits_toggled),
        )

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
print(f'{hits_above_put=:0.2e} {hits_below_put=:0.2e}')
print(f'{evts_above_put=} {evts_below_put=} {evts_both_put=}')
print(f'{evts_above_put+evts_below_put=}')

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

if False:
    import scipy.special
    x = np.arange(len(num_hits_per_adder_bits_set))
    y = num_hits_per_adder_bits_set / scipy.special.comb(x[-1], x)
    # y = num_hits_per_adder_bits_cleared / np.sum(num_events_per_ctr)
    plt.bar(x,y,width=1)
    plt.semilogy()
    plt.xlabel('number of adder lines set at each hit')

if False:
    import scipy.special
    x = np.arange(len(num_hits_per_adder_bits_cleared))
    y = num_hits_per_adder_bits_cleared / scipy.special.comb(x[-1], x)
    # y = num_hits_per_adder_bits_cleared / np.sum(num_events_per_ctr)
    plt.bar(x,y,width=1)
    plt.semilogy()
    plt.xlabel('number of adder lines cleared at each hit')

if False:
    import scipy.special
    x = np.arange(len(num_hits_per_adder_bits_toggled))
    y = num_hits_per_adder_bits_toggled / scipy.special.comb(x[-1], x)
    # y = num_hits_per_adder_bits_toggled / np.sum(num_events_per_ctr)
    plt.bar(x,y,width=1)
    plt.semilogy()
    plt.xlabel('number of adder lines toggled at each hit')

if False:
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

if False:
    import matplotlib.colors
    # num_events_per_pixel_recv /= np.sum(num_events_per_pixel_put, axis=0, keepdims=True)
    plt.imshow(num_events_per_pixel_recv, norm=matplotlib.colors.LogNorm())
    # plt.imshow(num_events_per_pixel_put, norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.title('Number of Crosstalk Events per Pixel (PuT)')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.show()
    exit()

# plt.axhline(0.5)
plt.ylabel('fraction X of events with crosstalk')
plt.show()