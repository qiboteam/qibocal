# -*- coding: utf-8 -*-
"""Adds global CLI options."""

import click
import os
import yaml


@click.group()
@click.argument("runcard", metavar="DEFAULT_CARD", type=click.Path(exists=True))
@click.argument("folder", type=click.Path())
@click.pass_context
def command(ctx, runcard, folder):
    ctx.ensure_object(dict)
    """qcvv: Quantum Calibration Verification and Validation using Qibo."""
    if os.path.exists(folder):
        raise (RuntimeError("Calibration folder with the same name already exists."))
    else:
        path = os.path.join(os.getcwd(), folder)
        click.echo(f"Creating directory {path}.")
        os.makedirs(path)
        ctx.obj["path"] = path

    with open(runcard, "r") as file:
        settings = yaml.safe_load(file)

    ctx.obj["nqubits"] = settings["nqubits"]
    backend = settings["backend"]

    from qibo import set_backend

    set_backend(backend)
