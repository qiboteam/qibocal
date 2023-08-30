"""Helper functions for the cli module"""
import datetime
import getpass
import json
import os
import shutil
from pathlib import Path

import yaml
from qibolab.serialize import dump_runcard

from qibocal.config import log, raise_error
from qibocal.utils import allocate_qubits_pairs, allocate_single_qubits

RUNCARD = "runcard.yml"
UPDATED_PLATFORM = "new_platform.yml"
PLATFORM = "platform.yml"
META = "meta.json"


def dump_report(meta, path):
    # update end time
    e = datetime.datetime.now(datetime.timezone.utc)
    meta["end-time"] = e.strftime("%H:%M:%S")
    (path / META).write_text(json.dumps(meta, indent=4))
    # report = ReportBuilder(path)
    # report.run()


def create_qubits_dict(runcard):
    """Qubits dictionary."""
    platform = runcard.platform_obj
    if platform is not None:
        if any(isinstance(i, list) for i in runcard.qubits):
            return allocate_qubits_pairs(platform, runcard.qubits)

        return allocate_single_qubits(platform, runcard.qubits)

    return runcard.qubits


def prepare_output(card, runcard, path):
    """Methods that takes care of:
    - dumping original platform
    - storing qq runcard
    - generating meta.yml
    """
    if runcard.backend == "qibolab":
        dump_runcard(runcard.platform_obj, path / PLATFORM)

    (path / RUNCARD).write_text(yaml.dump(card))

    import qibocal

    e = datetime.datetime.now(datetime.timezone.utc)
    meta = {}
    meta["title"] = path.name
    meta["backend"] = str(runcard.backend_obj)
    meta["platform"] = str(runcard.platform_obj)
    meta["date"] = e.strftime("%Y-%m-%d")
    meta["start-time"] = e.strftime("%H:%M:%S")
    meta["end-time"] = e.strftime("%H:%M:%S")
    meta["versions"] = runcard.backend_obj.versions  # pylint: disable=E1101
    meta["versions"]["qibocal"] = qibocal.__version__

    (path / META).write_text(json.dumps(meta, indent=4))

    return meta


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
    return Path(folder)
