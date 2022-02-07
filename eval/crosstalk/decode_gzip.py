#!/usr/bin/env python
import numpy as np
import gzip, base64
import sys

while True:
    try:
        data = input('enter base64 encded data: ')
        frames = np.frombuffer(
            gzip.decompress(
                base64.b64decode(data)
            ),
            dtype=np.uint8,
        )

        if frames.size == 2*24*24:
            shape = 24, 24
            frames = frames.reshape(2, 24, 24)
        elif frames.size == 2*48*48:
            shape = 48, 48
            frames = frames.reshape(2, 48, 48)
        else:
            print(f'invalid size {frames.size=}')
            exit(1)

        for pos in np.ndindex(*shape):
            pixel = frames[(...,) + pos]
            if pixel[0] != pixel[1]:
                print(f'{pos=} {pixel[0]} -> {pixel[1]}')
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(e)