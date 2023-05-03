import os
import shutil

from qibocal.plots.utils import get_data_subfolders


def test_get_data_subfolders():
    number_of_subfolders = 3
    parent_folder = "test_plots"
    folders = [f"{parent_folder}/subfolder{i}" for i in range(number_of_subfolders)]
    for folder in folders:
        if not os.path.isdir(folder):
            os.makedirs(folder)
    subfolders = get_data_subfolders(parent_folder)
    shutil.rmtree(parent_folder)
    assert len(subfolders) == number_of_subfolders
    for i in range(number_of_subfolders):
        assert f"subfolder{i}" in subfolders
