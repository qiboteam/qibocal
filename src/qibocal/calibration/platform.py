from dataclasses import dataclass
from pathlib import Path

from qibolab import Platform, create_platform, locate_platform

from .calibration import CALIBRATION, Calibration


@dataclass
class CalibrationPlatform(Platform):
    """Qibolab platform with calibration information."""

    calibration: Calibration = None
    """Calibration information."""

    @classmethod
    def from_platform(cls, platform: Platform):
        name = platform.name
        if name == "dummy":
            calibration = Calibration.model_validate_json(
                (Path(__file__).parent / "dummy.json").read_text()
            )
        else:
            path = locate_platform(name)
            calibration = Calibration.model_validate_json(
                (path / CALIBRATION).read_text()
            )
        # TODO: this is loading twice a platform
        return cls(**vars(platform), calibration=calibration)

    def dump(self, path: Path):
        super().dump(path)
        self.calibration.dump(path)


def create_calibration_platform(name: str) -> CalibrationPlatform:
    platform = create_platform(name)
    return CalibrationPlatform.from_platform(platform)