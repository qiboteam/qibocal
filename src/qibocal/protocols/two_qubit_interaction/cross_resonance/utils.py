from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId, QubitPairId
from qibolab.pulses import Pulse, Rectangular, PulseType, Gaussian
from qibolab.native import NativePulse
from qibolab.platform import Platform
from qibolab.qubits import QubitId, QubitPairId

STATES = ["I", "X"]
"""Setup states for the cross resonance gate calibration: {Identity, RX}."""

PROJECTIONS = ['Z', 'Y', 'X']
"""Standard projections for measurements."""


def ro_projection_pulse(platform: Platform, qubit, start=0, projection = PROJECTIONS[0]):
    """Create a readout pulse for a given qubit."""
    qd_pulse = platform.create_RX90_pulse(qubit, start=start)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=qd_pulse.finish)

    if projection == PROJECTIONS[0]:   
        qd_pulse.amplitude = 0
    elif projection == PROJECTIONS[1]:
        qd_pulse.relative_phase=0
    elif projection == PROJECTIONS[2]:
        qd_pulse.relative_phase=180
    else:
        raise ValueError(f"Invalid measurement <{projection}>")
    
    return qd_pulse, ro_pulse