"""Helper functions for the cli module"""

import datetime
import getpass
import os
import shutil

import yaml

from qibocal.config import log, raise_error


def load_yaml(path):
    """Load yaml file from disk."""
    with open(path) as file:
        data = yaml.safe_load(file)
    return data


def generate_output_folder(folder, force):
    """Generation of qq output folder
    Args:
        folder (path): path for the output folder. If None it will be created a folder automatically
        force (bool): option to overwrite the output folder if it exists already.

    Returns:
        Output path.
    """
    if folder is None:
        e = datetime.datetime.now()
        user = getpass.getuser().replace(".", "-")
        date = e.strftime("%Y-%m-%d")
        folder = f"{date}-{'000'}-{user}"
        num = 0
        while os.path.exists(folder):
            log.info(f"Directory {folder} already exists.")
            num += 1
            folder = f"{date}-{str(num).rjust(3, '0')}-{user}"
            log.info(f"Trying to create directory {folder}")
    elif os.path.exists(folder) and not force:
        raise_error(RuntimeError, f"Directory {folder} already exists.")
    elif os.path.exists(folder) and force:
        log.warning(f"Deleting previous directory {folder}.")
        shutil.rmtree(os.path.join(os.getcwd(), folder))

    path = os.path.join(os.getcwd(), folder)
    log.info(f"Creating directory {folder}.")
    os.makedirs(path)
    return folder
