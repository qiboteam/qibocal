from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.platforms.platform import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab.pulses import PulseSequence
from scipy.optimize import curve_fit

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.config import log
from qibocal.data import DataUnits

from . import utils

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

    sequence: Dict[List[PulseSequence], str]
    """Sequences ran on the experiment"""
    lenght: Dict[List[int], str]
    """Lenght of the sequences ran on the experiment"""
    probabilities: Dict[List[float], str]
    """Probabilities obtained for each sequence"""
    fitted_parameters: Dict[List[Tuple], List]
    """Raw fitted parameters."""


class StdRBData(DataUnits):
    """RabiAmplitude data acquisition."""

    def __init__(self):
        super().__init__(
            "data",
            {"sequence": "dimensionless", "probabilities": "dimensionless"},
            options=["qubit"],
        )


def _acquisition(
    params: StdRBParameters, platform: AbstractPlatform, qubits: Qubits
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
        sequences = rb_sequencer.get_sequences(qubit)
        for sequence in sequences.values():
            print(sequence[0])
            results = platform.execute_pulse_sequence(
                sequence,
                ExecutionParameters(
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    acquisition_type=AcquisitionType.DISCRIMINATION,
                    averaging_mode=AveragingMode.CYCLIC,
                ),
            )

            ro_pulses = sequence.ro_pulses

            # average msr, phase, i and q over the number of shots defined in the runcard
            result = results[ro_pulses[qubit].serial]
            r = result.serialize
            r.update(
                {
                    "sequence[dimensionless]": sequence,
                    "lenght[dimensionless]": len(sequence),
                    "probability[dimensionless]": results.state_0_probability,
                    "qubit": qubit,
                }
            )
            data.add_data_from_dict(r)

    return data


def _fit(data: StdRBData) -> StdRBResults:
    """Post-processing for Standard RB."""

    qubits = data.df["qubit"].unique()

    fitted_parameters = {}
    fidelities = {}

    for qubit in qubits:
        qubit_data = data.df[data.df["qubit"] == qubit]

        sequence_lenght = qubit_data["lenght"].pint.to("dimensionless").pint.magnitude
        probabilities = (
            qubit_data["probability"].pint.to("dimensionless").pint.magnitude
        )

        y_min = np.min(probabilities.values)
        y_max = np.max(probabilities.values)
        x_min = np.min(sequence_lenght.values)
        x_max = np.max(sequence_lenght.values)
        x = (sequence_lenght.values - x_min) / (x_max - x_min)
        y = (probabilities.values - y_min) / (y_max - y_min)

        pguess = [0.5, 0.9, 0.8]
        try:
            popt, pcov = curve_fit(utils.fit, x, y, p0=pguess, maxfev=100000)

            # TODO: Translate properly
            translated_popt = [
                y_min + (y_max - y_min) * popt[0],
                (y_max - y_min) * popt[1],
                popt[2] * (y_max - y_min) + y_max,
            ]
            fidelity = popt[1]

            # r=1−p−(1−p)/sequence_lenght  ???

        except:
            log.warning("rabi_fit: the fitting was not succesful")
            fidelity = 0
            fitted_parameters = [0] * 4

        fidelities[qubit] = fidelity
        fitted_parameters[qubit] = translated_popt

    return StdRBResults(fitted_parameters, fidelities)


def _plot(data: StdRBData, fit: StdRBResults, qubit):
    """Plotting function for Standard RB."""
    return utils.plot(data, fit, qubit)


StdRB = Routine(_acquisition, _fit, _plot)
"""Standard RB Routine object."""
