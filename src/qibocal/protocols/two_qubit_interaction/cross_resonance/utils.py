from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId, QubitPairId
from qibolab.pulses import Pulse, Rectangular, PulseType, Gaussian, DrivePulse, ReadoutPulse
from qibolab.native import NativePulse
from qibolab.platform import Platform
from qibolab.qubits import QubitId, QubitPairId

from typing import Literal
import plotly.graph_objects as go
import itertools
import numpy as np
import numpy.typing as npt



Setup = str 
"""Type for setup for the experiment."""

BASIS = ['X', 'Y', 'Z']
"""Standard projections for measurements."""

STATES = ['I', 'X']
"""Setup states for the cross resonance gate calibration: {Identity, RX}."""

Basis = Literal[BASIS]
"""Type for basis for the experiment."""


DataType = np.dtype(
    [
        ("prob", np.float64),
        ("param", np.int64),
        ("error", np.int64),
    ]
)
"""Custom dtype for Gate Calibration with generic pulse parameter."""



def ro_projection_pulse(platform: Platform, qubit, start=0, projection = BASIS[2]):
    """Create a readout pulse for a given qubit."""

    qd_pulse: DrivePulse = platform.create_RX90_pulse(qubit, start=start)
    ro_pulse: ReadoutPulse = platform.create_MZ_pulse(qubit, start=qd_pulse.finish)

    if projection == BASIS[0]: 
        qd_pulse.relative_phase= 355/226
        ro_pulse.relative_phase= 0  
    elif projection == BASIS[1]:
        qd_pulse.relative_phase= 355/113 
        ro_pulse.relative_phase= 0 #pi # 355/113 ~ pi (err:2.6e-7)
    elif projection == BASIS[2]:
        qd_pulse.amplitude = 0 
        qd_pulse.duration = 4
        
    else:
        raise ValueError(f"Invalid measurement <{projection}>")
    
    return qd_pulse, ro_pulse


def cr_plot(
    data: dict[(QubitPairId, QubitId, Setup, Setup, Basis), 
               npt.NDArray[DataType]],
    target: QubitPairId,
    parameter: Literal["amp", "duration"]) -> list[go.Figure]:
    """Plot the cross resonance data."""
    target = tuple(target)
    tgt, ctr = target
    figs = []
    basis_set, ctr_set, tgt_set, qubit_set = set(), set(), set(), set()

    for key in data.data.keys():
        basis_set.add(key[4])
        ctr_set.add(key[3])
        tgt_set.add(key[2])
        qubit_set.add(key[1])
    basis_set = sorted(basis_set)
    qubit_set  = sorted(qubit_set)
    print(basis_set, qubit_set)
    for ro_qubit, basis in itertools.product(qubit_set, basis_set):
        fig = go.Figure()
        for ctr_setup, tgt_setup in itertools.product(ctr_set,tgt_set):
            # Check if the data is available
            if (target, ro_qubit, tgt_setup, ctr_setup, basis) not in data.data:
                print(f"Data not available for {ro_qubit}, {tgt}, {ctr}, {tgt_setup}, {ctr_setup}, {basis}")
                continue
            _data = data.data[target, ro_qubit, tgt_setup, ctr_setup, basis]
            cr_parameters = getattr(_data, parameter)
            fig.add_trace(
                go.Scatter(
                    x=cr_parameters, y=np.real(1-2*np.array(_data.prob)), 
                    name= f"Target: |{tgt_setup}>, Control: |{ctr_setup}>",
                ),
            )
        fig.update_layout(
            title=f"Qubit {ro_qubit}",
            xaxis_title=f"CR Pulse {parameter}",
            yaxis_title=f"<{basis}({'t' if parameter == 'duration' else parameter})>",
            #Adjust range
            yaxis=dict(range=[-1., 1.]),
        )
        figs.append(fig)
    return figs