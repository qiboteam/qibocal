# -*- coding: utf-8 -*-
"""Adds global CLI options."""

import os
import shutil

import click
import yaml

from qcvv import calibrations
from qcvv.config import log, raise_error

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("platform", metavar="PLATFORM_NAME")
@click.argument("action_runcard", metavar="ACTION_CARD", type=click.Path(exists=True))
@click.argument("folder", type=click.Path())
@click.option("--force", is_flag=True)
def command(platform, action_runcard, folder, force=None):

    """qcvv: Quantum Calibration Verification and Validation using Qibo."""
    from qibo.backends import construct_backend

    platform = construct_backend("qibolab", platform=platform).platform

    if os.path.exists(folder) and not force:
        raise_error(
            RuntimeError, "Calibration folder with the same name already exists."
        )
    else:
        from qibolab.paths import qibolab_folder

        runcard = qibolab_folder / "runcards" / f"{platform}.yml"
        path = os.path.join(os.getcwd(), folder)
        if not force:
            log.info(f"Creating directory {path}.")
            os.makedirs(path)
        shutil.copy(runcard, f"{path}/")

    with open(action_runcard, "r") as file:
        action_settings = yaml.safe_load(file)

    platform.connect()
    platform.setup()
    platform.start()
    for routine_name in action_settings:
        routine = getattr(calibrations, routine_name)
        routine(
            platform,
            action_settings[routine_name]["qubit"],
            action_settings[routine_name]["settings"],
            path,
        )

    platform.stop()
    platform.disconnect()
