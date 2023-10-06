import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift

from .utils import nextpow2


def get_fft(img, return_type="complex"):
    h, w = img.shape
    h2, w2 = nextpow2(h), nextpow2(w)
    img_FFT = fft2(img, (h2, w2))
    img_FFT = fftshift(img_FFT)

    return_type = return_type.replace(" ", "").lower()
    return_type = "mag" if return_type == "magnitude" else return_type
    if return_type == "complex":
        return img_FFT
    elif return_type == "mag":
        return np.abs(img_FFT)
    elif return_type == "phase":
        return np.angle(img_FFT)
    elif return_type == "real":
        return np.real(img_FFT)
    elif return_type == "imag":
        return np.imag(img_FFT)
    else:
        raise ValueError("Invalid return type")


def get_ifft(img_FFT):
    img_FFT = ifftshift(img_FFT)
    img = ifft2(img_FFT)
    img = np.real(img)
    return img


def bwLP(d0, n, M, N=None):
    if N == None:
        N = M
    KM, KN = (M // 2, N // 2)
    u, v = np.mgrid[-KM:KM, -KN:KN]
    duv = np.hypot(u, v)
    H = 1 / np.sqrt(1 + (duv / d0) ** (2 * n))
    return fftshift(H)


def bwHP(d0, n, M, N=None):
    if N == None:
        N = M
    LP = bwLP(d0, n, M, N)
    return np.sqrt(1 - LP**2)


def bwBP(d1, n1, d2, n2, M, N=None):
    if N == None:
        N = M
    LP = bwLP(d2, n2, M, N)
    HP = bwHP(d1, n1, M, N)
    return LP * HP


def bwBS(d1, n1, d2, n2, M, N=None):
    if N == None:
        N = M
    BP = bwBP(d1, n1, d2, n2, M, N)
    return np.sqrt(1 - BP**2)
