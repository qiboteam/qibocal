# -*- coding: utf-8 -*-
"""Adds global CLI options."""
import base64
import pathlib
import shutil
import socket
import subprocess
import uuid
from importlib import import_module
from urllib.parse import urljoin

import click
from qibo.config import log, raise_error

from qibocal.cli.builders import ActionBuilder

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

# options for report upload
UPLOAD_HOST = (
    "qibocal@localhost"
    if socket.gethostname() == "saadiyat"
    else "qibocal@login.qrccluster.com"
)
TARGET_DIR = "qibocal-reports/"
ROOT_URL = "http://login.qrccluster.com:9000/"


# options for report compare
UPLOAD_COMPARE_HOST = (
    "david.fuentes@localhost"
    if socket.gethostname() == "saadiyat"
    else "david.fuentes@login.qrccluster.com"
)
TARGET_COMPARE_DIR = "qibocal-report-compare/"


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
    """

    builder = ActionBuilder(runcard, folder, force)
    builder.execute()
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
# @click.argument("output_folder", metavar="FOLDER", type=click.Path(exists=True))
@click.argument("folders", metavar="FOLDER", type=click.Path(exists=True), nargs=-1)
def compare(folders):
    """Uploads list of folders to be compared into server"""

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
        UPLOAD_COMPARE_HOST,
        "exit",
    )

    str_line = " ".join(repr(ele) for ele in ssh_command_line)

    log.info(f"Checking SSH connection to {UPLOAD_COMPARE_HOST}.")

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

    # upload data
    i = 0
    randname = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode()
    for name in foldernames:
        if name[-1] == "/":
            name = name[:-1]

        # subdirectory = get_data_subdirectory_name(name)
        origin_data_folder = f"{name}"
        newdir = TARGET_COMPARE_DIR + randname  # + f"/data_{i}"
        # newdir = TARGET_COMPARE_DIR + f"data_{i}"
        rsync_command = (
            "rsync",
            "-aLz",
            "--chmod=ug=rwx,o=rx",
            f"{origin_data_folder}",
            f"{UPLOAD_COMPARE_HOST}:{newdir}",
        )

        log.info(f"Uploading ({origin_data_folder}) to {UPLOAD_COMPARE_HOST}:{newdir}")
        try:
            subprocess.run(rsync_command, check=True)
        except subprocess.CalledProcessError as e:
            msg = f"Failed to upload output: {e}"
            raise RuntimeError(msg) from e

        # url = urljoin(ROOT_URL)
        log.info(f"Upload completed")
        i = +1


def folders_exists(folders):
    from glob import glob

    foldernames = []
    for foldername in folders:
        expanded = list(glob(foldername))
        if len(expanded) == 0 and "*" not in foldername:
            raise (click.BadParameter("file '{}' not found".format(foldername)))
        foldernames.extend(expanded)

    return foldernames


def check_folder_structure(folderList):
    import os

    all_subdirList = []
    for folder in folderList:
        folder_subdirList = []
        for dirName, subdirList, fileList in os.walk(folder):
            folder_subdirList.append(subdirList)
        all_subdirList.append(folder_subdirList)

    return all(x == all_subdirList[0] for x in all_subdirList)


def get_data_subdirectory_name(folder):
    import os

    folder_subdirList = []
    for dirName, subdirList, fileList in os.walk(folder):
        folder_subdirList.append(subdirList)
    return "".join(folder_subdirList[1])
