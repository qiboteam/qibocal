# -*- coding: utf-8 -*-
import re

import numpy as np

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

def line(x, p0, p1):
    # Slope                   : p[0]
    # Intercept               : p[1]
    return p0 * x + p1


def parse(key):
    name = key.split("[")[0]
    unit = re.search(r"\[([A-Za-z0-9_]+)\]", key).group(1)
    return name, unit


def G_f_d(x, p0, p1, p2):
    G = np.sqrt(np.cos(np.pi*(x-p0)*p1)**2+p2**2*np.sin(np.pi*(x-p0)*p1)**2)
    return np.sqrt(G)


def freq_r_transmon(x, p0, p1, p2, p3, p4, p5):
    return p5 + p4**2*G_f_d(x, p0, p1, p2)/(p5-p3*p5*G_f_d(x, p0, p1, p2))


def kordering(m,ng=0.4999):
    a1 = (round(2*ng+1/2)%2)*(round(ng)+1*(-1)**m *divmod(m+1,2)[0])
    a2 = (round(2*ng-1/2)%2)*(round(ng)-1*(-1)**m *divmod(m+1,2)[0])
    return a1 + a2


def mathieu(index,x):    
    if index < 0:
        dummy =  mathieu_b(-index,x)
    else:
        dummy =  mathieu_a(index,x)
    return dummy


def freq_q_mathieu(curr, curr0, xi, d, Ec, Ej, ng=0.499):
    index1 = int(2*(ng + kordering(1,ng)))
    index0 = int(2*(ng + kordering(0,ng)))         
    Ej = Ej*G_f_d(curr, curr0, xi ,d)
    return Ec * (mathieu(index1,-Ej/(2*Ec)) - mathieu(index0,-Ej/(2*Ec)))


def freq_r_mathieu(curr, f_h, g, curr0, xi, d, Ec, Ej, ng=0.499):
    G = G_f_d(curr, curr0, xi ,d)     
    f_q = freq_q_mathieu(curr, curr0, xi, d, Ec, Ej, ng)
    f_r = f_h + g**2*np.sqrt(G)/(f_h-f_q)
    return f_r