"""qibocal: Quantum Calibration Verification and Validation using Qibo."""

import os
from pathlib import Path

from . import protocols
from .auto.execute import Executor
from .calibration import create_calibration_platform
from .cli import command
from .version import __version__

__all__ = [
    "Executor",
    "protocols",
    "command",
    "__version__",
    "create_calibration_platform",
]

FOLDER = Path(__file__).parents[2] / "platforms"


def _add_platforms():
    new_folder = str(FOLDER)
    existing_platforms = os.getenv("QIBOLAB_PLATFORMS", "")
    platforms_set = {str(Path(p)) for p in existing_platforms.split(os.pathsep) if p}
    if new_folder not in platforms_set:
        platforms_set.add(new_folder)
        os.environ["QIBOLAB_PLATFORMS"] = os.pathsep.join(platforms_set)


_add_platforms()

DEFAULT_EXECUTOR = Executor.create(".routines", platform="dummy")
"""Default executor, registered as a qibocal submodule.

It is defined for streamlined usage of qibocal protocols in simple
contexts, where no additional options has to be defined for the
executor.

This is not meant to be used directly, thus is not meant to be publicly
exposed.
"""
