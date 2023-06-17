"""Adds global CLI options."""
import base64
import pathlib
import shutil
import socket
import subprocess
import uuid
from urllib.parse import urljoin

import click
import yaml
from qibo.config import log, raise_error

from ..cli.builders import AcquisitionBuilder, ActionBuilder, ReportBuilder

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

# options for report upload
UPLOAD_HOST = (
    "qibocal@localhost"
    if socket.gethostname() == "saadiyat"
    else "qibocal@login.qrccluster.com"
)
TARGET_DIR = "qibocal-reports/"
ROOT_URL = "http://login.qrccluster.com:9000/"


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
@click.option(
    "--update/--no-update",
    default=True,
    help="Use --no-update option to avoid updating iteratively the platform."
    "With this option the new runcard will not be produced.",
)
def command(runcard, folder, force, update):
    """qibocal: Quantum Calibration Verification and Validation using Qibo.

    Arguments:

     - RUNCARD: runcard with declarative inputs.
     - PLATFORM_RUNCARD: Qibolab's platform runcard. If not provided Qibocal will use the default one in
                         https://github.com/qiboteam/qibolab_platforms_qrc.
    """
    card = yaml.safe_load(pathlib.Path(runcard).read_text(encoding="utf-8"))
    builder = ActionBuilder(card, folder, force, update=update)
    builder.run()
    if update:
        builder.dump_platform_runcard()
    builder.dump_report()


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
def acquire(runcard, folder, force):
    """qibocal: Quantum Calibration Verification and Validation using Qibo.

    Arguments:

     - RUNCARD: runcard with declarative inputs.
     - PLATFORM_RUNCARD: Qibolab's platform runcard. If not provided Qibocal will use the default one in
                         https://github.com/qiboteam/qibolab_platforms_qrc.
    """
    card = yaml.safe_load(pathlib.Path(runcard).read_text(encoding="utf-8"))
    builder = AcquisitionBuilder(card, folder, force)
    builder.run()


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("folder", metavar="folder", type=click.Path(exists=True))
def report(folder):
    builder = ReportBuilder(pathlib.Path(folder))
    builder.run()


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("output_folder", metavar="FOLDER", type=click.Path(exists=True))
def upload(output_folder):
    """Uploads output folder to server"""

    output_path = pathlib.Path(output_folder)

    # check the rsync command exists.
    if not shutil.which("rsync"):
        raise_error(
            RuntimeError,
            "Could not find the rsync command. Please make sure it is installed.",
        )

    # check that we can authentica with a certificate
    ssh_command_line = (
        "ssh",
        "-o",
        "PreferredAuthentications=publickey",
        "-q",
        UPLOAD_HOST,
        "exit",
    )

    str_line = " ".join(repr(ele) for ele in ssh_command_line)

    log.info(f"Checking SSH connection to {UPLOAD_HOST}.")

    try:
        subprocess.run(ssh_command_line, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            (
                "Could not validate the SSH key. "
                "The command\n%s\nreturned a non zero exit status. "
                "Please make sure that your public SSH key is on the server."
            )
            % str_line
        ) from e
    except OSError as e:
        raise RuntimeError(
            "Could not run the command\n{}\n: {}".format(str_line, e)
        ) from e

    log.info("Connection seems OK.")

    # upload output
    randname = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode()
    newdir = TARGET_DIR + randname

    rsync_command = (
        "rsync",
        "-aLz",
        "--chmod=ug=rwx,o=rx",
        f"{output_path}/",
        f"{UPLOAD_HOST}:{newdir}",
    )

    log.info(f"Uploading output ({output_path}) to {UPLOAD_HOST}")
    try:
        subprocess.run(rsync_command, check=True)
    except subprocess.CalledProcessError as e:
        msg = f"Failed to upload output: {e}"
        raise RuntimeError(msg) from e

    url = urljoin(ROOT_URL, randname)
    log.info(f"Upload completed. The result is available at:\n{url}")
