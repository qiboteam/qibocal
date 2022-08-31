# -*- coding: utf-8 -*-
import re

import numpy as np


def lorenzian(frequency, amplitude, center, sigma, offset):
    # http://openafox.com/science/peak-function-derivations.html
    return (amplitude / np.pi) * (
        sigma / ((frequency - center) ** 2 + sigma**2)
    ) + offset


def parse(key):
    name = key.split("[")[0]
    unit = re.search(r"\[([A-Za-z0-9_]+)\]", key).group(1)
    return name, unit
