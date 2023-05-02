import re

import numpy as np
import pandas as pd
import pint
from scipy.special import mathieu_a, mathieu_b


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
    return p0 + p1 * np.sin(x * p2 + p3) * np.exp(-x * p4)


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


def line(x, p0, p1):
    # Slope                   : p[0]
    # Intercept               : p[1]
    return p0 * x + p1


def parse(key):
    name = key.split("[")[0]
    unit = re.search(r"\[([A-Za-z0-9_]+)\]", key).group(1)
    return name, unit


def G_f_d(x, p0, p1, p2):
    # Current offset:          : p[0]
    # 1/I_0, Phi0=Xi*I_0       : p[1]
    # Junction asymmetry d     : p[2]
    G = np.sqrt(
        np.cos(np.pi * (x - p0) * p1) ** 2
        + p2**2 * np.sin(np.pi * (x - p0) * p1) ** 2
    )
    return np.sqrt(G)


def freq_r_transmon(x, p0, p1, p2, p3, p4, p5):
    # Current offset:                                      : p[0]
    # 1/I_0, Phi0=Xi*I_0                                   : p[1]
    # Junction asymmetry d                                 : p[2]
    # f_q0/f_rh, f_q0 = Qubit frequency at zero flux       : p[3]
    # Qubit-resonator coupling at zero magnetic flux, g_0  : p[4]
    # High power resonator frequency, f_rh                 : p[5]
    return p5 + p4**2 * G_f_d(x, p0, p1, p2) / (p5 - p3 * p5 * G_f_d(x, p0, p1, p2))


def kordering(m, ng=0.4999):
    # Ordering function sorting the eigenvalues |m,ng> for the Schrodinger equation for the
    # Cooper pair box circuit in the phase basis.
    a1 = (round(2 * ng + 1 / 2) % 2) * (round(ng) + 1 * (-1) ** m * divmod(m + 1, 2)[0])
    a2 = (round(2 * ng - 1 / 2) % 2) * (round(ng) - 1 * (-1) ** m * divmod(m + 1, 2)[0])
    return a1 + a2


def mathieu(index, x):
    # Mathieu's characteristic value a_index(x).
    if index < 0:
        dummy = mathieu_b(-index, x)
    else:
        dummy = mathieu_a(index, x)
    return dummy


def freq_q_mathieu(x, p0, p1, p2, p3, p4, p5=0.499):
    # Current offset:                                      : p[0]
    # 1/I_0, Phi0=Xi*I_0                                   : p[1]
    # Junction asymmetry d                                 : p[2]
    # Charging energy E_C                                  : p[3]
    # Josephson energy E_J                                 : p[4]
    # Effective offset charge ng                           : p[5]
    index1 = int(2 * (p5 + kordering(1, p5)))
    index0 = int(2 * (p5 + kordering(0, p5)))
    p4 = p4 * G_f_d(x, p0, p1, p2)
    return p3 * (mathieu(index1, -p4 / (2 * p3)) - mathieu(index0, -p4 / (2 * p3)))


def freq_r_mathieu(x, p0, p1, p2, p3, p4, p5, p6, p7=0.499):
    # High power resonator frequency, f_rh                 : p[0]
    # Qubit-resonator coupling at zero magnetic flux, g_0  : p[1]
    # Current offset:                                      : p[2]
    # 1/I_0, Phi0=Xi*I_0                                   : p[3]
    # Junction asymmetry d                                 : p[4]
    # Charging energy E_C                                  : p[5]
    # Josephson energy E_J                                 : p[6]
    # Effective offset charge ng                           : p[7]
    G = G_f_d(x, p2, p3, p4)
    f_q = freq_q_mathieu(x, p2, p3, p4, p5, p6, p7)
    f_r = p0 + p1**2 * np.sqrt(G) / (p0 - f_q)
    return f_r


def pint_to_float(x):
    if isinstance(x, pd.Series):
        return x.apply(pint_to_float)
    elif isinstance(x, pint.Quantity):
        return x.to(x.units).magnitude
    else:
        return x
