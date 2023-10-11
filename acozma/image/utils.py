import numpy as np


def min_max_scale(image):
    min_val, max_val = np.min(image), np.max(image)
    return (image - min_val) / (max_val - min_val)


def nextpow2(N):
    n = 1
    while n < N:
        n *= 2
    return n
