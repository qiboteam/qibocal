import os
from colorsys import hls_to_rgb


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
