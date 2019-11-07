#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Random generation tools."""
from secrets import randbits

import numpy as np


def generate_random_bit_array(bits_count):
    """Generates cryptographically random binary number."""
    number = randbits(bits_count)

    bits = bin(number)[2:]  # Cut `0b` prefix.

    bits = [int(x) for x in bits]

    result = np.zeros(bits_count)

    for idx in range(len(bits) - 1, -1, -1):
        result[idx] = bits[idx]

    return result
