import re

import numpy as np


def parse(key):
    name = key.split("[")[0]
    unit = re.search(r"\[([A-Za-z0-9_]+)\]", key).group(1)
    return name, unit


def landscape(x, p0, p1, p2):
    #
    # Amplitude                     : p[0]
    # Offset                        : p[1]
    # Phase offset                  : p[2]
    return np.sin(x + p2) * p0 + p1
