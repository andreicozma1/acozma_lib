import numpy as np


def print_info(image, start=None):
    info = f"min: {np.min(image):.3f} | max: {np.max(image):.3f}"
    print(f"{start or '':>10}: {info}")
    return info


def nextpow2(N):
    n = 1
    while n < N:
        n *= 2
    return n
