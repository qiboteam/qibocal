"""SWAP experiment for two qubit gates, chevron plot."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from sklearn.preprocessing import minmax_scale

from qibocal.auto.operation import (
    DATAFILE,
    Data,
    Parameters,
    Protocol,
    QubitPairId,
    Results,
)
from qibocal.calibration import CalibrationPlatform, calibration
from qibocal.config import log
from qibocal.protocols.utils import HZ_TO_GHZ, table_dict, table_html
from qibocal.protocols import update
from qibocal.protocols.randomized_benchmarking.utils import (
    IndexedCircuit,
    IndexedResult,)
from qibo import Circuit, gates

__all__ = ["standard_error_amplification"]

@dataclass
class StandardErrorAmplificationParameters(Parameters):
    """Parameters for the standard error amplification protocol."""

    depths: list
    """List of number of CZ gates for the error amplification."""

    native: Literal["CZ"] = "CZ"
    """Two qubit interaction to be calibratred. Only CZ is supported for now."""

    nshots: int = 10
    """Number of shots for each depth."""


@dataclass
class StandardErrorAmplificationData(Data):
    """Data for the standard error amplification protocol."""


def generate_sea_circuit(
    params: StandardErrorAmplificationParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> list[Circuit]:

    """Build the Standard Error Amplification (SEA) circuit for CZ conditional-phase calibration. Applies 2n total CZ gates.

    Args:
        params: Parameters for the standard error amplification protocol.
        platform: Calibration platform.
        targets: List of qubit pairs to run the protocol on.
    
    Returns:
        List of circuits for each target qubit pair.
    """

    circuits: list[Circuit] = []

    for depth in params.depths:
        for target in targets:
            circuit = Circuit(2)
            circuit.add(gates.RX(q=target[0], theta=np.pi / 2))
            circuit.add(gates.CZ(q0=target[0], q1=target[1]))

            for _ in range(2* depth - 1):
                circuit.add(gates.RX(q=target[0], theta=np.pi))
                circuit.add(gates.RY(q=target[1], theta=np.pi))
                circuit.add(gates.CZ(q0=target[0], q1=target[1]))
            
            circuit.add(gates.RX(q=target[0], theta=np.pi / 2))
            circuit.add(gates.M(q=target[0]))
        
        circuits.append(circuit)

def _sea_circuit_acquisition(
    params: StandardErrorAmplificationParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> StandardErrorAmplificationData:
    """Run the Standard Error Amplification (SEA) circuit for CZ conditional-phase calibration. Applies 2n total CZ gates.

    Args:
        params: Parameters for the standard error amplification protocol.
        platform: Calibration platform.
        targets: List of qubit pairs to run the protocol on.

    """


@dataclass
class StandardErrorAmplificationResults(Results):
    """Results for the standard error amplification protocol."""


def _acquisition(
    params: StandardErrorAmplificationParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> StandardErrorAmplificationData:
    """Run run for the standard error amplification protocol.
    
    Args:
        params: Parameters for the standard error amplification protocol.
        platform: Calibration platform.
        targets: List of qubit pairs to run the protocol on.
        
    Returns:
        Data for the standard error amplification protocol.
    """
    data = _sea_circuit_acquisition(params, platform, targets)
    return data


def _fit(
    data: StandardErrorAmplificationData,
) -> StandardErrorAmplificationResults:
    """Fit the data for the standard error amplification protocol.
    
    Args:
        data: Data for the standard error amplification protocol."""

def _plot(
    results: StandardErrorAmplificationResults,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> go.Figure:
    """Plot the results for the standard error amplification protocol.
    
    Args:
        results: Results for the standard error amplification protocol.
        platform: Calibration platform.
        targets: List of qubit pairs to plot the results for.
    """

def _update(
    results: StandardErrorAmplificationResults,
    platform: CalibrationPlatform,
    target: QubitPairId,
):
    """Update the calibration platform with the results of the standard error amplification protocol.
    
    Args:
        results: Results for the standard error amplification protocol.
        platform: Calibration platform.
        target: Qubit pair to update the calibration for.
    """
    target = tuple(target)
    platform.calibration.two_qubits[target].conditional_phase = results.conditional_phase[target]


sea = Protocol(_acquisition, _fit, _plot, _update)
