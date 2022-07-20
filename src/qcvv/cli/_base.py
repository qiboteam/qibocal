# -*- coding: utf-8 -*-
"""Adds global CLI options."""

import os
import shutil

import click
import yaml

from qcvv.config import log, raise_error

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("platform", metavar="PLATFORM_NAME")
@click.argument("action_runcard", metavar="ACTION_CARD", type=click.Path(exists=True))
@click.argument("folder", type=click.Path())
@click.option("--force", type=bool)
def command(platform, action_runcard, folder, force=None):

    """qcvv: Quantum Calibration Verification and Validation using Qibo."""
    from qibo.backends import GlobalBackend, set_backend

    set_backend("qibolab", platform=platform)

    platform = GlobalBackend().platform

    if os.path.exists(folder) and force is None:
        raise_error(
            RuntimeError, "Calibration folder with the same name already exists."
        )
    else:
        from qibolab.paths import qibolab_folder

        runcard = qibolab_folder / "runcards" / f"{platform}.yml"
        path = os.path.join(os.getcwd(), folder)
        if force is None:
            log.info(f"Creating directory {path}.")
            os.makedirs(path)
        shutil.copy(runcard, f"{path}/")

    with open(action_runcard, "r") as file:
        action_settings = yaml.safe_load(file)

    platform.connect()
    platform.setup()
    platform.start()
    for routine in action_settings:
        if routine == "resonator_spectroscopy_attenuation":
            from qcvv.resonator_spectroscopy import resonator_spectroscopy_attenuation

            resonator_spectroscopy_attenuation(
                platform,
                action_settings[routine]["qubit"],
                action_settings[routine]["settings"],
                path,
            )
        else:
            raise_error(ValueError, "Unknown calibration routine")

    platform.stop()
    platform.disconnect()
