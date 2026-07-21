from dataclasses import dataclass
from pathlib import Path

from qibolab import Parameters, Platform, create_platform, locate_platform
from qibolab._core.dummy.platform import create_dummy
from qibolab._core.platform.platform import PARAMETERS

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
                if natives[q].RX is None or len(natives[q].RX) == 0
                else natives[q].RX[0][1].relative_phase == 0.0
            )
            phase_rx90 = (
                True
                if natives[q].RX90 is None or len(natives[q].RX90) == 0
                else natives[q].RX90[0][1].relative_phase == 0.0
            )
            phase_rx12 = (
                True
                if natives[q].RX12 is None or len(natives[q].RX12) == 0
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
        try:
            calibration = Calibration.model_validate_json(
                (path / CALIBRATION).read_text()
            )
        except FileNotFoundError:
            calibration = Calibration()
        # TODO: this is loading twice a platform
        return cls(**vars(platform), calibration=calibration)

    @classmethod
    def from_datafolder(
        cls, folder_path: Path, platform_name: str, dummy_hardware: bool
    ):
        """Create a calibration platform from a serialized data folder.

        The platform is rebuilt from the configuration saved in the experiment history,
        using the ``parameters.json`` and ``calibration.json`` files stored in the data folder
        rather than the platform definition. If a ``platform_name`` is provided, the hardware
        configuration is loaded from that platform; otherwise, a dummy hardware
        configuration is used so that acquisition-related fields are still present
        without requiring a live instrument setup.
        """

        parameters = Parameters.model_validate_json(
            (folder_path / PARAMETERS).read_text()
        )

        calibration = Calibration.model_validate_json(
            (folder_path / CALIBRATION).read_text()
        )

        platform = (
            create_dummy() if dummy_hardware is None else create_platform(platform_name)
        )
        platform.parameters = parameters
        platform.name = platform_name

        return cls(
            calibration=calibration,
            **vars(platform),
        )

    def dump(self, path: Path):
        super().dump(path)
        self.calibration.dump(path)


def create_calibration_platform(name: str) -> CalibrationPlatform:
    """This function builds a ``CalibrationPlatform`` object which is sentitive of the hardware,
    so it needs information about the clusters and its connection. Has to be used for acquisition.
    """
    platform = create_platform(name)
    return CalibrationPlatform.from_platform(platform)
