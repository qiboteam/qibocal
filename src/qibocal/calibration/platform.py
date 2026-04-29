from dataclasses import dataclass
from pathlib import Path

from qibolab import Platform, create_platform, locate_platform

from .calibration import CALIBRATION, Calibration

__all__ = ["CalibrationPlatform", "create_calibration_platform"]


class CalibrationError(Exception):
    def __init__(self, *args):
        super().__init__(*args)


@dataclass
class CalibrationPlatform(Platform):
    """Qibolab platform with calibration information."""

    calibration: Calibration = None
    """Calibration information."""

    def __post_init__(self):
        """Post-initialization method for the Platform class.

        Validates that all X rotation native gates (RX, RX90, RX12) for each qubit
        have a relative_phase of 0.0. If any gate does not meet this condition,
        logs an error and raises a ValueError.
        """

        natives = self.parameters.native_gates.single_qubit
        for q in self.qubits:
            phase_rx = (
                True
                if natives[q].RX is None
                else natives[q].RX[0][1].relative_phase == 0.0
            )
            phase_rx90 = (
                True
                if natives[q].RX90 is None
                else natives[q].RX90[0][1].relative_phase == 0.0
            )
            phase_rx12 = (
                True
                if natives[q].RX12 is None
                else natives[q].RX12[0][1].relative_phase == 0.0
            )

            if not (phase_rx and phase_rx90 and phase_rx12):
                raise CalibrationError(
                    "All X rotation must be set with relative_phase = 0."
                )

    @classmethod
    def from_platform(cls, platform: Platform):
        name = platform.name
        path = locate_platform(name)
        calibration = Calibration.model_validate_json((path / CALIBRATION).read_text())
        # TODO: this is loading twice a platform
        return cls(**vars(platform), calibration=calibration)

    def dump(self, path: Path):
        super().dump(path)
        self.calibration.dump(path)


def create_calibration_platform(name: str) -> CalibrationPlatform:
    platform = create_platform(name)
    return CalibrationPlatform.from_platform(platform)
