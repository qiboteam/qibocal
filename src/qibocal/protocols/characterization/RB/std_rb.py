from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from scipy.optimize import curve_fit

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.config import log
from qibocal.data import DataUnits
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
    nshots: int
    """Number of shots."""
    relaxation_time: float
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


class StdRBData(DataUnits):
    """RabiAmplitude data acquisition."""

    def __init__(self):
        super().__init__(
            "data",
            {
                "sequence": "dimensionless",
                "length": "dimensionless",
                "probabilities": "dimensionless",
            },
            options=["qubit"],
        )


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

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include qubit drive pulse amplitude
    data = StdRBData()

    # sweep the parameter
    for qubit in qubits.values():
        sequences, circuits = rb_sequencer.get_sequences(qubit.name)
        for sequence, circuit in zip(sequences.values(), circuits.values()):
            results = platform.execute_pulse_sequence(
                sequence[0],
                ExecutionParameters(
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    acquisition_type=AcquisitionType.DISCRIMINATION,
                    averaging_mode=AveragingMode.CYCLIC,
                ),
            )

            ro_pulses = sequence[0].ro_pulses
            # average msr, phase, i and q over the number of shots defined in the runcard

            for ro_pulse in ro_pulses:
                result = results[ro_pulse.serial]
                r = result.serialize

                r.update(
                    {
                        "sequence[dimensionless]": 0,  # TODO: Store sequences
                        "length[dimensionless]": len(circuit[0]),
                        "probabilities[dimensionless]": r["0"][0],
                        "qubit": qubit.name,
                    }
                )
                data.add_data_from_dict(r)

    return data


def _fit(data: StdRBData) -> StdRBResults:
    """Post-processing for Standard RB."""

    qubits = data.df["qubit"].unique()

    fitted_parameters = {}
    fidelities_primitive = {}
    fidelities = {}
    average_errors_gate = {}

    for qubit in qubits:
        qubit_data = data.df[data.df["qubit"] == qubit]

        sequence_length = qubit_data["length"].pint.to("dimensionless").pint.magnitude
        probabilities = (
            qubit_data["probabilities"].pint.to("dimensionless").pint.magnitude
        )

        x = sequence_length.values
        y = probabilities.values

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


StdRB = Routine(_acquisition, _fit, _plot)
"""Standard RB Routine object."""
