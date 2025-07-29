from ...auto.operation import QubitId
from ...calibration import CalibrationPlatform


def _get_lo_frequency(platform: CalibrationPlatform, qubit: QubitId) -> float:
    """Get LO frequency given QubitId.

    Currently it assumes that instruments with LOs is first one.
    """
    probe = platform.channels[platform.qubits[qubit].probe]
    lo_config = platform.config(probe.lo)
    return lo_config.frequency
