import json
import os
import pathlib
import shutil

from ..config import raise_error
from .utils import META, UPDATED_PLATFORM


def update(path: pathlib.Path):
    """Perform copy of updated platform in QIBOLAB_PLATFORM."

    Arguments:
        - input_path: Qibocal output folder.
    """

    new_platform_path = path / UPDATED_PLATFORM

    if not new_platform_path.exists():
        raise_error(FileNotFoundError, f"No updated runcard platform found in {path}.")

    platform_name = json.loads((path / META).read_text())["platform"]
    platform_path = pathlib.Path(os.getenv("QIBOLAB_PLATFORMS")) / platform_name

    for filename in os.listdir(new_platform_path):
        shutil.copy(
            new_platform_path / filename,
            platform_path / filename,
        )