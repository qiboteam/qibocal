from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId, QubitPairId
from qibolab.pulses import Pulse, Rectangular, PulseType, Gaussian, DrivePulse, ReadoutPulse
from qibolab.native import NativePulse
from qibolab.platform import Platform
from qibolab.qubits import QubitId, QubitPairId

from numpy import pi
STATES = ["I", "X"]
"""Setup states for the cross resonance gate calibration: {Identity, RX}."""

BASIS = ['X', 'Y', 'Z']
"""Standard projections for measurements."""


def ro_projection_pulse(platform: Platform, qubit, start=0, projection = BASIS[0]):
    """Create a readout pulse for a given qubit."""
    qd_pulse: DrivePulse = platform.create_RX90_pulse(qubit, start=start)
    ro_pulse: ReadoutPulse = platform.create_qubit_readout_pulse(qubit, start=qd_pulse.finish)

    if projection == BASIS[0]:   
        qd_pulse.amplitude = 0
    elif projection == BASIS[1]:
        qd_pulse.relative_phase=0
    elif projection == BASIS[2]:
        qd_pulse.relative_phase= pi # 355/113 ~ pi (err:1e-7)
    else:
        raise ValueError(f"Invalid measurement <{projection}>")
    
    return qd_pulse, ro_pulse