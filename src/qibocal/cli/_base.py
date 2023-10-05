"""Adds global CLI options."""
import pathlib

import click
import yaml

from .acquisition import acquire as acquisition
from .autocalibration import autocalibrate
from .fit import fit as fitting
from .report import report as reporting
from .upload import upload_report

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group()
def command():
    """Welcome to Qibocal!
    Qibo module to calibrate and characterize self-hosted QPUs.
    """


@command.command(context_settings=CONTEXT_SETTINGS)
@click.argument(
    "runcard", metavar="RUNCARD", type=click.Path(exists=True, path_type=pathlib.Path)
)
@click.option(
    "folder",
    "-o",
    type=click.Path(path_type=pathlib.Path),
    help="Output folder. If not provided a standard name will generated.",
)
@click.option(
    "force",
    "-f",
    is_flag=True,
    help="Use --force option to overwrite the output folder.",
)
@click.option(
    "--update/--no-update",
    default=True,
    help="Use --no-update option to avoid updating iteratively the platform."
    "With this option the new runcard will not be produced.",
)
def auto(runcard, folder, force, update):
    """Autocalibration

    Arguments:

     - RUNCARD: runcard with declarative inputs.
    """
    card = yaml.safe_load(runcard.read_text(encoding="utf-8"))
    autocalibrate(card, folder, force, update)


@command.command(context_settings=CONTEXT_SETTINGS)
@click.argument(
    "runcard", metavar="RUNCARD", type=click.Path(exists=True, path_type=pathlib.Path)
)
@click.option(
    "folder",
    "-o",
    type=click.Path(path_type=pathlib.Path),
    help="Output folder. If not provided a standard name will generated.",
)
@click.option(
    "force",
    "-f",
    is_flag=True,
    help="Use --force option to overwrite the output folder.",
)
def acquire(runcard, folder, force):
    """Data acquisition

    Arguments:

     - RUNCARD: runcard with declarative inputs.
    """
    card = yaml.safe_load(runcard.read_text(encoding="utf-8"))
    acquisition(card, folder, force)


@command.command(context_settings=CONTEXT_SETTINGS)
@click.argument(
    "folder", metavar="folder", type=click.Path(exists=True, path_type=pathlib.Path)
)
def report(folder):
    """Report generation

    Arguments:

    - FOLDER: input folder.

    """
    reporting(folder)


@command.command(context_settings=CONTEXT_SETTINGS)
@click.argument(
    "folder", metavar="folder", type=click.Path(exists=True, path_type=pathlib.Path)
)
@click.option(
    "--update/--no-update",
    default=True,
    help="Use --no-update option to avoid updating iteratively the platform."
    "With this option the new runcard will not be produced.",
)
def fit(folder: pathlib.Path, update):
    """Post-processing analysis

    Arguments:

    - FOLDER: input folder.

    """
    fitting(folder, update)


@command.command(context_settings=CONTEXT_SETTINGS)
@click.argument(
    "path", metavar="FOLDER", type=click.Path(exists=True, path_type=pathlib.Path)
)
@click.option(
    "--tag",
    default=None,
    type=str,
    help="Optional tag.",
)
def upload(path, tag):
    """Uploads output folder to server

    Arguments:

    - FOLDER: input folder.
    """
    upload_report(path, tag)
