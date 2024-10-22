import os
from dataclasses import dataclass
from pathlib import Path

from qibolab import Platform, create_platform

from .calibration import Calibration


@dataclass
class CalibrationPlatform:
    """Qibolab platform with calibration information."""

    platform: Platform = None
    """Qibolab platforms."""
    calibration: Calibration = None
    """Calibration information."""

    @property
    def natives(self):
        return self.platform.natives

    @property
    def parameters(self):
        return self.platform.parameters

    def execute(self, *args, **kwargs):
        return self.execute(*args, **kwargs)

    @classmethod
    def load(cls, name: str):

        platform = create_platform(name)
        path = Path(os.getenv("QIBOLAB_PLATFORMS")) / name
        calibration = Calibration.model_validate_json(path.read_text())

        return cls(platform, calibration)


# def create_calibration_platform(name: str) -> CalibrationPlatform:

#     platform = create_platform(name)
#     calibration = Calibration.
