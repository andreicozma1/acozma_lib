import math

import numpy as np


def get_histogram_stats(h, N):
    n0 = np.cumsum(h)
    n1 = n0[N - 1] - n0

    mu0 = np.zeros(N)
    mu1 = np.zeros(N)

    var0 = np.zeros(N)
    var1 = np.zeros(N)

    A, B = 0, 0
    for i in range(0, N):
        A += i * h[i]
        B += i**2 * h[i]

        if n0[i] == 0:
            mu0[i] = 0.0
            var0[i] = 0.0
        else:
            mu0[i] = np.int32(A / n0[i] + 0.5)
            var0[i] = (B - (A**2) / n0[i]) / n0[i]

    A, B = 0, 0
    for i in reversed(range(0, N - 1)):
        A += (i + 1) * h[i + 1]
        B += (i + 1) ** 2 * h[i + 1]

        if n1[i] == 0:
            mu1[i] = 0.0
            var1[i] = 0.0
        else:
            mu1[i] = np.int32(A / n1[i] + 0.5)
            var1[i] = (B - (A**2) / n1[i]) / n1[i]

    return [n0, n1], [mu0, mu1], [var0, var1]


def mean(n, mu, var):
    return mu[0][-1]


def median(n, mu, var):
    i = 0
    while n[0][i] < n[0][-1] / 2:
        i += 1
    return i


def middle(n, mu, var):
    i_min = 0
    while 0 == n[0][i_min]:
        i_min += 1

    N = n[0].size
    i_max = N - 1
    while 0 == n[0][i_max]:
        i_max -= 1

    return (i_min + i_max) // 2


def isodata(n, mu, var):
    i0, i1 = np.int32(mu[0][-1]), -1

    while i0 != i1:
        if n[0][i0] == 0 or n[1][i0] == 0:
            break

        i1 = i0
        i0 = np.int32(mu[0][i0] + mu[1][i0]) // 2

    return i0


def otsu(n, mu, var):
    n_sum = n[0][-1]
    s2b_max = 0
    i_max = 0

    N = n[0].size
    for i in range(0, N):
        P0 = n[0][i] / n_sum
        P1 = n[1][i] / n_sum
        s2b = P0 * P1 * (mu[0][i] - mu[1][i]) ** 2

        if s2b_max <= s2b:
            s2b_max = s2b
            i_max = i

    return i_max


def bayesian(n, mu, var):
    n_sum = n[0][-1]
    var_eps = 1 / 12
    E_min = 1.0e38
    i_min = 0

    N = n[0].size
    for i in range(0, N - 2):
        if n[0][i] == 0 or n[1][i] == 0:
            continue

        P0 = n[0][i] / n_sum
        P1 = n[1][i] / n_sum

        E0 = P0 * math.log(var[0][i] + var_eps) - 2 * P0 * math.log(P0)
        E1 = P1 * math.log(var[1][i] + var_eps) - 2 * P1 * math.log(P1)

        if E_min > E0 + E1:
            E_min = E0 + E1
            i_min = i

    return i_min
