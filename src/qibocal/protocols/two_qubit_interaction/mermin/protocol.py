from dataclasses import dataclass
from typing import Optional

from qibolab import ExecutionParameters
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, Parameters, Results, Routine

from ...readout_mitigation_matrix import readout_mitigation_matrix
from ...utils import calculate_frequencies
from .circuits import create_mermin_circuits
from .pulses import create_mermin_sequences
from .utils import compute_mermin

READOUT_BASIS = [["X", "X", "Y"], ["X", "Y", "X"], ["Y", "X", "X"], ["Y", "Y", "Y"]]


@dataclass
class MerminParameters(Parameters):
    """Mermin experiment input parameters."""

    native: Optional[bool] = False
    """If True a circuit will be created using only GPI2 and CZ gates."""
    apply_error_mitigation: Optional[bool] = False
    """Error mitigation model"""


@dataclass
class MerminData(Data):
    """Mermin acquisition data."""


@dataclass
class MerminResults(Results):
    """Mermin results."""


def _acquisition_pulses(
    params: MerminParameters,
    platform: Platform,
    targets: list[QubitId],
) -> MerminData:
    r"""Data acquisition for CHSH protocol using pulse sequences."""

    if params.apply_error_mitigation:
        mitigation_data, _ = readout_mitigation_matrix.acquisition(
            readout_mitigation_matrix.parameters_type.load(
                dict(pulses=True, nshots=params.nshots)
            ),
            platform,
            [targets],
        )

        mitigation_results, _ = readout_mitigation_matrix.fit(mitigation_data)

    mermin_sequences = create_mermin_sequences(
        platform, targets, readout_basis=READOUT_BASIS
    )
    options = ExecutionParameters(nshots=params.nshots)

    mermin_frequencies = []
    for sequence in mermin_sequences:
        results = platform.execute_pulse_sequence(sequence, options=options)
        frequencies = calculate_frequencies(results, targets)
        mermin_frequencies.append(frequencies)

    mermin_bare = compute_mermin(frequencies=mermin_frequencies)
    print(mermin_bare)

    return MerminData()


def _acquisition_circuits(
    params: MerminParameters,
    platform: Platform,
    targets: list[tuple[QubitId]],
) -> MerminData:
    r"""Data acquisition for CHSH protocol using pulse sequences."""

    if params.apply_error_mitigation:
        mitigation_data, _ = readout_mitigation_matrix.acquisition(
            readout_mitigation_matrix.parameters_type.load(
                dict(pulses=False, nshots=params.nshots)
            ),
            platform,
            [targets],
        )

        mitigation_results, _ = readout_mitigation_matrix.fit(mitigation_data)

    mermin_circuits = create_mermin_circuits(
        targets, native=params.native, readout_basis=READOUT_BASIS
    )

    mermin_frequencies = []
    for circuit in mermin_circuits:
        results = circuit(nshots=params.nshots)
        mermin_frequencies.append(results.frequencies())

    mermin_bare = compute_mermin(frequencies=mermin_frequencies)

    return MerminData()


def _fit(data: MerminData) -> MerminResults:
    return MerminResults()


def _plot(
    data: MerminData, fit: MerminResults, target: tuple[QubitId, QubitId, QubitId]
):

    return [], ""


mermin_pulses = Routine(_acquisition_pulses, _fit, _plot)
mermin_circuits = Routine(_acquisition_circuits, _fit, _plot)
