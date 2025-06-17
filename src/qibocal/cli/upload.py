"""Upload report to server."""

import base64
import json
import pathlib
import shutil
import socket
import subprocess
import uuid
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urljoin

from qibo.config import log, raise_error

from qibocal.auto.output import META, Metadata


class UploadLab(Enum):
    """Lab where the report is uploaded."""

    TII = "tii"
    S14 = "s14"


@dataclass
class UploadConfig:
    """Configuration for the upload process."""

    host: str
    """Host to upload the report to."""
    target_dir: str
    """Target directory on the host where the report will be uploaded."""
    root_url: str
    """Root URL for accessing the uploaded report."""

    @staticmethod
    def from_lab(lab_name: str) -> "UploadConfig":
        try:
            lab = getattr(UploadLab, lab_name.upper())
        except KeyError:
            raise ValueError(f"Unknown lab: {lab_name}")
        if lab == UploadLab.TII:
            return UploadConfig(
                host=(
                    "qibocal@saadiyat"
                    if socket.gethostname() in ("saadiyat", "dalma")
                    else "qibocal@login.qrccluster.com"
                ),
                target_dir="qibocal-reports/",
                root_url="http://login.qrccluster.com:9000/",
            )
        elif lab == UploadLab.S14:
            return UploadConfig(
                host=("qibocal_user@10.246.80.226"),
                target_dir="qibocal-reports/",
                root_url="http://10.246.80.226:9000/",
            )


def upload_report(path: pathlib.Path, tag: str, author: str, lab: str) -> str:
    # load meta and update tag
    meta = Metadata.load(path)
    meta.author = author
    meta.tag = tag
    (path / META).write_text(json.dumps(meta.dump(), indent=4), encoding="utf-8")
    config = UploadConfig.from_lab(lab)
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
        config.host,
        "exit",
    )

    str_line = " ".join(repr(ele) for ele in ssh_command_line)

    log.info(f"Checking SSH connection to {config.host}.")

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
    newdir = config.target_dir + randname

    rsync_command = (
        "rsync",
        "-aLz",
        "--chmod=ug=rwx,o=rx",
        f"{path}/",
        f"{config.host}:{newdir}",
    )

    log.info(f"Uploading output ({path}) to {config.host}")
    try:
        subprocess.run(rsync_command, check=True)
    except subprocess.CalledProcessError as e:
        msg = f"Failed to upload output: {e}"
        raise RuntimeError(msg) from e

    url = urljoin(config.root_url, randname)
    log.info(f"Upload completed. The result is available at:\n{url}")
    return url
