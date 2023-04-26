import os
from colorsys import hls_to_rgb

import numpy as np
import pandas as pd

from qibocal.data import DataUnits


def get_data_subfolders(folder):
    # iterate over multiple data folders
    subfolders = []
    for file in os.listdir(folder):
        d = os.path.join(folder, file)
        if os.path.isdir(d):
            subfolders.append(os.path.basename(d))

    return subfolders[::-1]


def get_color(number):
    return "rgb" + str(hls_to_rgb((0.75 - number * 3 / 20) % 1, 0.4, 0.75))


def get_color_state0(number):
    return "rgb" + str(hls_to_rgb((-0.35 - number * 9 / 20) % 1, 0.6, 0.75))


def get_color_state1(number):
    return "rgb" + str(hls_to_rgb((-0.02 - number * 9 / 20) % 1, 0.6, 0.75))


def load_data(folder, subfolder, routine, data_format, name):
    file = f"{folder}/{subfolder}/{routine}/{name}.csv"
    data = DataUnits()
    all_columns = pd.read_csv(file, nrows=1).columns.tolist()
    if "fit" in name or "parameters" in name:
        data.df = pd.read_csv(file, header=[0], index_col=[0])

    else:
        data.df = pd.read_csv(file, header=[0], skiprows=[1], index_col=[0])

    return data
