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

    fidelities: Dict[List[float], List]
    """Probabilities obtained for each sequence"""
    fitted_parameters: Dict[List[Tuple], List]
    """Raw fitted parameters."""


class StdRBData(DataUnits):
    """RabiAmplitude data acquisition."""

    def __init__(self):
        super().__init__(
            "data",
            {
                "sequence": "dimensionless",
                "lenght": "dimensionless",
                "probabilities": "dimensionless",
            },
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
            result = results[ro_pulses[0].serial]
            r = result.serialize

            print(circuit[0])

            r.update(
                {
                    # "sequence[dimensionless]": [int(x) for x in circuit[0]],
                    "sequence[dimensionless]": 420,
                    "lenght[dimensionless]": len(circuit[0]),
                    "probabilities[dimensionless]": r["state_0"][0],
                    "qubit": qubit.name,
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
            qubit_data["probabilities"].pint.to("dimensionless").pint.magnitude
        )

        # TODO: Translate no
        x = sequence_lenght.values
        y = probabilities.values

        pguess = [0.5, 0.9, 0.8]
        try:
            popt, pcov = curve_fit(
                utils.RB_fit, x, y, p0=pguess, maxfev=100000
            )  # TODO: bounds on p [0,1] for A and B

            # TODO: remove translate

            translated_popt = popt
            p = popt[1]

            fidelity = 1 - ((1 - p) / 2)
            # Divide by magic number
            magic_number = 1.875

            fitted_parameters[qubit] = translated_popt
            fidelities[qubit] = fidelity
            print("sucess", fitted_parameters)

        except:
            log.warning("RB_fit: the fitting was not succesful")
            fidelities[qubit] = 0
            fitted_parameters[qubit] = [0] * 3
            print("failed", fitted_parameters)

    return StdRBResults(fidelities, fitted_parameters)


def _plot(data: StdRBData, fit: StdRBResults, qubit):
    """Plotting function for Standard RB."""
    return utils.plot(data, fit, qubit)


StdRB = Routine(_acquisition, _fit, _plot)
"""Standard RB Routine object."""
