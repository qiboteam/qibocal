# -*- coding: utf-8 -*-
"""Adds global CLI options."""
import click

from qcvv import calibrations
from qcvv.cli.builders import ActionBuilder
from qcvv.config import log, raise_error

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("runcard", metavar="RUNCARD", type=click.Path(exists=True))
@click.option(
    "folder",
    "-o",
    type=click.Path(),
    help="Output folder. If not provided a standard name will generated.",
)
@click.option(
    "force",
    "-f",
    is_flag=True,
    help="Use --force option to overwrite the output folder.",
)
def command(runcard, folder, force=None):

    """qcvv: Quantum Calibration Verification and Validation using Qibo.

    Arguments:

     - RUNCARD: runcard with declarative inputs.
    """

    action_builder = ActionBuilder(runcard, folder, force)
    action_builder.execute()


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "port",
    "-p",
    "--port",
    default=8050,
    type=int,
    help="Localhost port to launch dash server.",
)
@click.option(
    "debug",
    "-d",
    "--debug",
    is_flag=True,
    help="Launch server in debugging mode.",
)
def live_plot(port, debug):
    """Real time plotting of calibration data on a dash server."""
    import socket

    from qcvv.web.app import app

    # change port if it is already used
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                break
        port += 1

    app.run_server(debug=debug, port=port)
