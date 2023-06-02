from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np
import plotly.graph_objects as go
from qibolab import ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import DataUnits
from qibocal.plots.utils import get_color

from collections import defaultdict

def calculate_probabilities(result1, result2):
    """Calculates two-qubit outcome probabilities from individual shots."""
    probs = {f"probability{v}": 0.0 for v in ["00", "01", "10", "11"]}
    shots = np.stack([result1.samples, result2.samples]).T.astype(int)
    values, counts = np.unique(shots, axis=0, return_counts=True)
    nshots = np.sum(counts)
    for (v1, v2), c in zip(values, counts):
        probs[f"probability{v1}{v2}"] = c / nshots
    return probs



@dataclass
class TomographyParameters(Parameters):
    """Tomography runcard inputs."""


    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""
    sequence: Optional[list] = None


@dataclass
class TomographyResults(Results):
    """Tomography outputs."""
    fitted_parameters: Dict[Union[str, int], Dict[str, float]]


class TomographyData(DataUnits):
    """Tomography acquisition outputs."""

    def __init__(self):
        super().__init__(
            name="data",
            quantities={
                        "rotation1": "dimensionless",
                        "rotation2": "dimensionless",
                        "probability00": "dimensionless",
                        "probability01": "dimensionless",
                        "probability10": "dimensionless",
                        "probability11": "dimensionless",
                        },
            options=["qubit"],
        )

def _acquisition(
    params: TomographyParameters,
    platform: Platform,
    qubits: Qubits,
) -> TomographyData:


    """State tomography for two qubits.

    The pulse sequence applied consists of two steps.
    First a state preperation sequence given by the user is applied, which
    prepares the target state. Then one additional pulse may be applied to each
    qubit to rotate the measurement basis.
    Following arXiv:0903.2030, tomography is performed by measuring in 15 different
    basis, which are defined by rotating using all pairs of I, RX90, RY90 and RX
    except (RX, RX).

    An example action runcard for using this routine is the following:

        platform: my_platform_name

        qubits: [1, 2]

        format: csv

        actions:

        state_tomography:
            sequence:
                # [[Pulse Type, Target Qubit]]
                # pulses given in the same row are played in parallel
                - [["RX", 1]]
                - [["RY90", 1], ["RY90", 2]]
                - [["CZ", [1, 2]]]
                - [["RY90", 1]]
            nshots: 50000

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (dict): Dict of target Qubit objects to perform the action
        sequence (list): List describing the pulse sequence to be used for state preperation.
            See example for more details.
        nshots (int): Number of shots to perform for each measurement.
    """
    if len(qubits) != 2:
        raise NotImplementedError("Tomography is only implemented for two qubits.")
    
    
    data = TomographyData()
    
    rotation_pulses = {
        "I": None,
        "RX": lambda qubit, start, phase: platform.create_RX_pulse(
            qubit, start, relative_phase=phase
        ),
        "RY": lambda qubit, start, phase: platform.create_RX_pulse(
            qubit, start, relative_phase=phase + np.pi / 2
        ),
        "RX90": lambda qubit, start, phase: platform.create_RX90_pulse(
            qubit, start, relative_phase=phase
        ),
        "RY90": lambda qubit, start, phase: platform.create_RX90_pulse(
            qubit, start, relative_phase=phase + np.pi / 2
        ),
    }
    tomography_basis = ["I", "RY90", "RX90"]

    qubit1, qubit2 = min(qubits.keys()), max(qubits.keys())

    sequence = params.sequence
    
    for label1 in tomography_basis:
        for label2 in tomography_basis:
            if not (label1 == "RX" and label2 == "RX"):
                phases = defaultdict(int)
                total_sequence = PulseSequence()
                # state preperation sequence
                for moment in sequence:
                    start = total_sequence.finish
                    for pulse_description in moment:
                        pulse_type, qubit = pulse_description[:2]
                        if pulse_type == "CZ":
                            cz_sequence, phases = platform.create_CZ_pulse_sequence(
                                qubit, start=total_sequence.finish
                            )
                            total_sequence = total_sequence + cz_sequence
                        elif pulse_type in rotation_pulses:
                            total_sequence.add(
                                rotation_pulses[pulse_type](qubit, start, phases[qubit])
                            )
                            phases[qubit] = 0
                        else:
                            raise NotImplementedError(f"Unknown gate {pulse_type}.")

                # basis rotation sequence
                start = total_sequence.finish
                if label1 != "I":
                    total_sequence.add(
                        rotation_pulses[label1](qubit1, start, phases[qubit1])
                    )
                if label2 != "I":
                    total_sequence.add(
                        rotation_pulses[label2](qubit2, start, phases[qubit2])
                    )
                # measurements
                start = total_sequence.finish
                measure1 = platform.create_MZ_pulse(qubit1, start=start)
                # measure2 = platform.create_MZ_pulse(qubit2, start=start)
                total_sequence.add(measure1)
                # total_sequence.add(measure2)

                results1 = platform.execute_pulse_sequence(total_sequence, ExecutionParameters(nshots=params.nshots))

                print(results1[measure1.serial].serialize)
                
                measure2 = platform.create_MZ_pulse(qubit2, start=start)
                total_sequence.add(measure2)
                total_sequence.remove(measure1)
                results2 = platform.execute_pulse_sequence(total_sequence, ExecutionParameters(nshots=params.nshots))

                
                
                
                print(results2[measure2.serial].serialize)
                
                # store the results
                r = calculate_probabilities(
                    results1[measure1.serial], results2[measure2.serial]
                )
                
                r["rotation1"] = label1
                r["rotation2"] = label2
                
                data.add_data_from_dict(r)

    return data
                
                
def _fit(data: TomographyData) -> TomographyResults:
    r"""Post-processing function for Tomography. """
    
    return TomographyResults(fitted_parameters = {0 : 0})

def _plot(data: TomographyData, fit: TomographyResults, qubit):
    """Plotting function for tomography."""
    
    figures = []
    fig = go.Figure()

    fitting_report = ""
    
    return figures, fitting_report

tomography = Routine(_acquisition, _fit, _plot)
"""tomography Routine  object."""