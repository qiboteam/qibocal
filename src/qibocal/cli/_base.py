"""Adds global CLI options."""
import base64
import os
import pathlib
import shutil
import socket
import subprocess
import uuid
from datetime import datetime
from glob import glob
from urllib.parse import urljoin

import click
from qibo.config import log, raise_error

from ..cli.auto_builder import AutoCalibrationBuilder
from ..cli.builders import ActionBuilder, load_yaml
from .utils import check_folder_structure, folders_exists, update_meta, update_runcard

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

# options for report upload
UPLOAD_HOST = (
    "qibocal@localhost"
    if socket.gethostname() == "saadiyat"
    else "qibocal@login.qrccluster.com"
)
TARGET_DIR = "qibocal-reports/"
ROOT_URL = "http://login.qrccluster.com:9000/"

TARGET_COMPARE_DIR = "qq-compare/"


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
    """qibocal: Quantum Calibration Verification and Validation using Qibo.

    Arguments:

     - RUNCARD: runcard with declarative inputs.

     - PLATFORM_RUNCARD: Qibolab's platform runcard. If not provided Qibocal will use the runcard available in Qibolab for the platform chosen.
    """

    builder = ActionBuilder(runcard, folder, force)
    builder.execute()


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
def autocalibration(runcard, folder, force=None):
    """qibocal: Quantum Calibration Verification and Validation using Qibo.

    Arguments:

     - RUNCARD: runcard with declarative inputs.

     - PLATFORM_RUNCARD: Qibolab's platform runcard. If not provided Qibocal will use the runcard available in Qibolab for the platform chosen.
    """

    builder = AutoCalibrationBuilder(runcard, folder, force)
    builder.run()
    builder.dump_platform_runcard()
    builder.dump_report()


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

    from qibocal.web.app import app

    # change port if it is already used
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                break
        port += 1

    app.run_server(debug=debug, port=port)


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


# qq-compare
@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("folders", metavar="FOLDER", type=click.Path(exists=True), nargs=-1)
def compare(folders):
    """Creates a comparision folder given N different data folders following qq-live data structure to be visualized

    Args:
        folder1 (string): absolute or relative path of data folder 1 to be compared
        ...
        folderN (string): absolute or relative path of data folder N to be compared
    """

    # check list of folders exists.
    foldernames = folders_exists(folders)

    # check folder structures matches
    if not check_folder_structure(foldernames):
        raise_error(
            RuntimeError,
            "Could not compare the list of folders. Please make sure that they have the same structure",
        )
    else:
        log.info(f"Folders are comparable.")

    # move old compare folder and remove
    now = str(datetime.today())
    tmp_folder = "qq-compare_" + now.replace(" ", "_")
    if os.path.isdir(TARGET_COMPARE_DIR):
        os.rename(TARGET_COMPARE_DIR, tmp_folder)

    os.mkdir(TARGET_COMPARE_DIR)
    # TODO: prepare meta.yml and runcard.yml as mix of all meta and runcard files
    shutil.copy(pathlib.Path(foldernames[0]).joinpath("meta.yml"), TARGET_COMPARE_DIR)
    shutil.copy(
        pathlib.Path(foldernames[0]).joinpath("runcard.yml"), TARGET_COMPARE_DIR
    )

    for i, folder in enumerate(foldernames):
        newdir = TARGET_COMPARE_DIR + f"data{i}"
        log.info(f"Copying ({folder}) into {newdir}")
        try:
            shutil.copytree(pathlib.Path(folder).joinpath("data"), newdir)
            for file in glob(os.path.join(folder, "*.html")):
                shutil.copy2(file, newdir)
            for file in glob(os.path.join(folder, "*.yml")):
                shutil.copy2(file, newdir)
        except subprocess.CalledProcessError as e:
            msg = f"Failed to upload output: {e}"
            raise RuntimeError(msg) from e

        if i > 0:
            # update meta.yml for comparing report
            metadata_new = load_yaml(os.path.join(folder, "meta.yml"))
            metadata = load_yaml(os.path.join(TARGET_COMPARE_DIR, "meta.yml"))
            update_meta(metadata, metadata_new)
            # update runcard.yml for comparing report
            rundata_new = load_yaml(os.path.join(folder, "runcard.yml"))
            rundata = load_yaml(os.path.join(TARGET_COMPARE_DIR, "runcard.yml"))
            update_runcard(rundata, rundata_new, TARGET_COMPARE_DIR)

    log.info(f"Upload completed")

    from qibocal.web.report import create_report

    create_report(TARGET_COMPARE_DIR)
    log.info(f"HTML report generated")
