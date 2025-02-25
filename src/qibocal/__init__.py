"""qibocal: Quantum Calibration Verification and Validation using Qibo."""

import os
from pathlib import Path

from . import protocols
from .auto.execute import Executor
from .auto.platforms.fake.platform import FOLDER
from .cli import command
from .version import __version__

__all__ = ["Executor", "protocols", "command", "__version__"]


def _add_fake():
    new_folder = str(FOLDER.parent)
    existing_platforms = os.getenv("QIBOLAB_PLATFORMS", "")
    platforms_set = {
        str(Path(p).resolve()) for p in existing_platforms.split(os.pathsep) if p
    }
    # Only add if not already present
    if new_folder not in platforms_set:
        platforms_set.add(new_folder)
        updated_platforms = os.pathsep.join(platforms_set)
        os.environ["QIBOLAB_PLATFORMS"] = updated_platforms


_add_fake()

DEFAULT_EXECUTOR = Executor.create(".routines", platform="fake")
"""Default executor, registered as a qibocal submodule.

It is defined for streamlined usage of qibocal protocols in simple
contexts, where no additional options has to be defined for the
executor.

This is not meant to be used directly, thus is not meant to be publicly
exposed.
"""
