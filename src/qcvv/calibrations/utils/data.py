# -*- coding: utf-8 -*-
import glob
import os
import pathlib
from turtle import update

import numpy as np
import qibolab
import yaml
from genericpath import isdir
from qibo.config import log, raise_error
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from scipy.optimize import curve_fit

from qcvv import plots
from qcvv.data import Dataset
from qcvv.decorators import plot


def get_latest_datafolder():
    cwd = pathlib.Path()
    list_dir = sorted(glob.glob(os.path.join(cwd, "*/")), key=os.path.getmtime)
    for i in range(len(list_dir)):
        if os.path.isdir(cwd / list_dir[-i] / "data"):
            return cwd / list_dir[-i]
