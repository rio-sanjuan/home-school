```python
# Check out the wikipedia page for an interesting read!
# https://en.wikipedia.org/wiki/Fast_inverse_square_root
# https://www.youtube.com/watch?v=p8u_k2LIZyo&t=11s

import struct, math


# From: https://stackoverflow.com/a/14431225/39182
def float2bits(f):
    """Convert a float to an int with the same bits."""
    s = struct.pack(">f", f)
    return struct.unpack(">l", s)[0]


def bits2float(b):
    """Reinterpret the bits of an int as a float."""
    s = struct.pack(">l", b)
    return struct.unpack(">f", s)[0]


# Transcribed from the original C code:
# https://en.wikipedia.org/wiki/Fast_inverse_square_root
def Q_rsqrt(number):
    threehalfs = 1.5

    x2 = number * 0.5
    y = number
    i = float2bits(y)  # evil floating point bit level hacking
    i = 0x5F3759DF - (i >> 1)  # what the fuck?
    y = bits2float(i)
    y = y * (threehalfs - (x2 * y * y))  # 1st iteration
    # y = y * (threehalfs - (x2 * y * y)) # 2nd iteration, this can be removed

    return y


def rsqrt(number):
    """A native "exact" x^{-1/2}"""
    return 1 / math.sqrt(number)


def compare(number):
    """Get the absolute error of the fast inverse square root function relative to an exact baseline."""
    return abs(rsqrt(number) - Q_rsqrt(number))


if __name__ == "__main__":
    print(f"Input: {500}, abs err: {compare(500.0)}")
    print(f"Input: {0.001}, abs err: {compare(0.001)}")
```