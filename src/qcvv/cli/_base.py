# -*- coding: utf-8 -*-
"""Adds global CLI options."""

import os

import click
import yaml

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument(
    "platform_runcard", metavar="PLATFORM_CARD", type=click.Path(exists=True)
)
@click.argument("action_runcard", metavar="ACTION_CARD", type=click.Path(exists=True))
@click.argument("folder", type=click.Path())
def command(platform_runcard, action_runcard, folder):

    """qcvv: Quantum Calibration Verification and Validation using Qibo."""
    from qibo.backends import set_backend, GlobalBackend
    platform = "tiiq"
    set_backend("qibolab", platform="tiiq", runcard=platform_runcard)

    platform = GlobalBackend().platform
    if os.path.exists(folder):
        raise (RuntimeError("Calibration folder with the same name already exists."))
    else:
        path = os.path.join(os.getcwd(), folder)
        click.echo(f"Creating directory {path}.")
        os.makedirs(path)

    with open(platform_runcard, "r") as file:
        platform_settings = yaml.safe_load(file)

    with open(action_runcard, "r") as file:
        action_settings = yaml.safe_load(file)

    
    for routine in action_settings:
        print(routine)
        if routine == "resonator_spectroscopy":
            from qcvv.res_spectr import resonator_spectroscopy
            resonator_spectroscopy(platform, action_settings[routine]["qubit"], action_settings[routine]["settings"], path)
