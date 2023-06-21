from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.qubits import QubitId
from scipy.optimize import curve_fit

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.config import log
from qibocal.protocols.characterization.RB import utils

"""
For info check
https://forest-benchmarking.readthedocs.io/en/latest/examples/randomized_benchmarking.html
"""


@dataclass
class StdRBParameters(Parameters):
    """Standard RB runcard inputs."""

    min_depth: int
    """Minimum depth."""
    max_depth: int
    """Minimum amplitude multiplicative factor."""
    step_depth: int
    """Minimum amplitude multiplicative factor."""
    runs: int
    """Number of random sequences per depth"""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class StdRBResults(Results):
    """Standard RB outputs."""

    fidelities: Dict[List[float], List]
    """Fidelity after magic number"""
    fidelities_primitive: Dict[List[float], List]
    """Primitive for fidelity after magic number"""
    average_errors_gate: Dict[List[float], List]
    """Error per average gate as a percentage"""
    fitted_parameters: Dict[List[Tuple], List]
    """Raw fitted parameters."""


StdRBType = np.dtype(
    [
        ("sequence", np.int64),
        ("length", np.int64),
        ("probabilities", np.float64),
    ]
)


@dataclass
class StdRBData(Data):
    """StdRB data acquisition."""

    data: dict[QubitId, npt.NDArray[StdRBType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, sequence, length, prob):
        """Store output for single qubit."""
        ar = np.empty((1,), dtype=StdRBType)
        ar["sequence"] = sequence
        ar["length"] = length
        ar["probabilities"] = prob
        if qubit in self.data:
            self.data[qubit] = np.rec.array(np.concatenate((self.data[qubit], ar)))
        else:
            self.data[qubit] = np.rec.array(ar)


def _acquisition(
    params: StdRBParameters, platform: Platform, qubits: Qubits
) -> StdRBData:
    r"""
    Data acquisition for Rabi experiment sweeping amplitude.
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse amplitude
    to find the drive pulse amplitude that creates a rotation of a desired angle.
    """

    # create sequences of pulses for the experiment

    depths = list(range(params.min_depth, params.max_depth, params.step_depth))

    rb_sequencer = utils.RBSequence(
        platform,
        depths,
        params.runs,
    )

    # create a Data object to store the results
    data = StdRBData()

    # sweep the parameter
    for qubit in qubits.values():
        sequences, circuits, ro_pulses = rb_sequencer.get_sequences_list(qubit.name)
        results = platform.execute_pulse_sequences(
            sequences,
            ExecutionParameters(
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.DISCRIMINATION,
                averaging_mode=AveragingMode.CYCLIC,
            ),
        )

        counts_depths = {}
        for depth in depths:
            starts = [pulse.start for pulse in ro_pulses[depth]]
            counts = {}
            for pulse in ro_pulses[depth]:
                counts[pulse] = starts.count(pulse.start)
            counts_depths[depth] = counts

        # average msr, phase, i and q over the number of shots defined in the runcard
        j = 0
        for depth in depths:
            for ro_pulse in counts_depths[depth].keys():
                for i in range(counts_depths[depth][ro_pulse]):
                    probs = results[ro_pulse.serial][i].probability(0)
                    qubit = ro_pulse.qubit
                    data.register_qubit(qubit, 0, len(circuits[j]), probs)
                    j += 1

    return data


def _fit(data: StdRBData) -> StdRBResults:
    """Post-processing for Standard RB."""

    qubits = data.qubits

    fitted_parameters = {}
    fidelities_primitive = {}
    fidelities = {}
    average_errors_gate = {}

    for qubit in qubits:
        qubit_data = data[qubit]

        x = qubit_data.length
        y = qubit_data.probabilities

        a_guess = np.max(y) - np.mean(y)
        p_guess = 0.9
        b_guess = np.mean(y)

        pguess = [a_guess, p_guess, b_guess]
        try:
            popt, pcov = curve_fit(
                utils.RB_fit,
                x,
                y,
                p0=pguess,
                maxfev=100000,
                # bounds=((-np.inf, 0, -np.inf), (np.inf, 1, np.inf)),
                bounds=((0, 0, 0), (1, 1, 1)),
            )

            p = popt[1]
            fidelity_primitive = 1 - ((1 - p) / 2)
            # Divide infidelity by magic number
            magic_number = 1.875
            infidelity = (1 - p) / magic_number
            fidelity = 1 - infidelity
            average_error_gate = infidelity * 100

            fitted_parameters[qubit] = popt
            fidelities[qubit] = fidelity
            fidelities_primitive[qubit] = fidelity_primitive
            average_errors_gate[qubit] = average_error_gate

        except:
            log.warning("RB_fit: the fitting was not succesful")
            fidelities[qubit] = 0.0
            fidelities_primitive[qubit] = 0.0
            average_errors_gate[qubit] = 0.0
            fitted_parameters[qubit] = [0] * 3

    return StdRBResults(
        fidelities, fidelities_primitive, average_errors_gate, fitted_parameters
    )


def _plot(data: StdRBData, fit: StdRBResults, qubit):
    """Plotting function for Standard RB."""
    return utils.plot(data, fit, qubit)


StdRB_unrolling = Routine(_acquisition, _fit, _plot)
"""Standard RB Routine object."""
