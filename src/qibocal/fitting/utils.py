import re

import numpy as np


def lorenzian(frequency, amplitude, center, sigma, offset):
    # http://openafox.com/science/peak-function-derivations.html
    return (amplitude / np.pi) * (
        sigma / ((frequency - center) ** 2 + sigma**2)
    ) + offset


def rabi(x, p0, p1, p2, p3, p4):
    # A fit to Superconducting Qubit Rabi Oscillation
    #   Offset                       : p[0]
    #   Oscillation amplitude        : p[1]
    #   Period    T                  : 1/p[2]
    #   Phase                        : p[3]
    #   Arbitrary parameter T_2      : 1/p[4]
    return p0 + p1 * np.sin(2 * np.pi * x * p2 + p3) * np.exp(-x * p4)


def ramsey(x, p0, p1, p2, p3, p4):
    # A fit to Superconducting Qubit Rabi Oscillation
    #   Offset                       : p[0]
    #   Oscillation amplitude        : p[1]
    #   DeltaFreq                    : p[2]
    #   Phase                        : p[3]
    #   Arbitrary parameter T_2      : 1/p[4]
    return p0 + p1 * np.sin(2 * np.pi * x * p2 + p3) * np.exp(-x * p4)


def exp(x, *p):
    return p[0] - p[1] * np.exp(-1 * x * p[2])


def flipping(x, p0, p1, p2, p3):
    # A fit to Flipping Qubit oscillation
    # Epsilon?? shoule be Amplitude : p[0]
    # Offset                        : p[1]
    # Period of oscillation         : p[2]
    # phase for the first point corresponding to pi/2 rotation   : p[3]
    return np.sin(x * 2 * np.pi / p2 + p3) * p0 + p1


def cos(x, p0, p1, p2, p3):
    # Offset                  : p[0]
    # Amplitude               : p[1]
    # Period                  : p[2]
    # Phase                   : p[3]
    return p0 + p1 * np.cos(2 * np.pi * x / p2 + p3)


def parse(key):
    name = key.split("[")[0]
    unit = re.search(r"\[([A-Za-z0-9_]+)\]", key).group(1)
    return name, unit
