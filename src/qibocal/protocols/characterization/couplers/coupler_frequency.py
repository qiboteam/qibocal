from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from qibolab.executionparameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import DataUnits


@dataclass
class CouplerFrequencyParameters(Parameters):
    """CouplerFrequency runcard inputs."""

    frequency_width: float
    """Frequency width."""
    frequency_step: float
    """Frequency step."""
    nshots: Optional[int] = None
    """Number of shots per point."""
    coupler_frequency: Optional[float] = None
    """Coupler frequency."""
    coupler_drive_duration: Optional[float] = None
    """Coupler drive duration."""
    coupler_drive_amplitude: Optional[float] = None
    """Coupler drive amplitude."""
    relaxation_time: Optional[float] = None
    """Relaxation time."""


@dataclass
class CouplerFrequencyResults(Results):
    """CouplerFrequency outputs when fitting will be done."""


class CouplerFrequencyData(DataUnits):
    """CouplerFrequency acquisition outputs."""

    def __init__(self):
        super().__init__(
            name="data",
            quantities={"frequency": "Hz"},
            options=[
                "coupler",
                "qubit",
                "probability",
            ],
        )


def _aquisition(
    params: CouplerFrequencyParameters,
    platform: AbstractPlatform,
    qubits: Qubits,
) -> CouplerFrequencyData:
    r"""
    Perform a spectrocopy experiment on the coupler.

    Args:
        platform: Platform to use.
        params: Experiment parameters.
        qubits: Qubits to use.

    Returns:
        DataUnits: Acquisition data.
    """
    # define the qubit to measure
    # FIXME: make general for multiple qubits
    if 2 in qubits:
        qubits.pop(2)
    qubit_pairs = list([qubit, 2] for qubit in qubits)

    # create a sequence
    sequence = PulseSequence()
    ro_pulses = {}
    cd_pulses = {}
    qd_pulses = {}

    # FIXME: add coupler pulses in the runcard
    if params.coupler_drive_duration is None:
        params.coupler_drive_duration = 4000
    if params.coupler_frequency is None:
        params.coupler_frequency = 3_600_000_000
    if params.coupler_drive_amplitude is None:
        params.coupler_drive_amplitude = 0.5
    for pair in qubit_pairs:
        cd_pulses[pair[0]] = platform.create_qubit_drive_pulse(
            pair[0], start=0, duration=params.coupler_drive_duration
        )
        qd_pulses[pair[1]] = platform.create_qubit_drive_pulse(
            pair[1], start=0, duration=params.coupler_drive_duration
        )
        cd_pulses[pair[0]].amplitude = params.coupler_drive_amplitude
        cd_pulses[pair[0]].frequency = params.coupler_frequency

        # FIXME: This should be done in the driver
        platform.qubits[pair[0]].drive_frequency = params.coupler_frequency
        platform.qubits[pair[0]].drive.local_oscillator.frequency = (
            params.coupler_frequency - params.frequency_width * 1.1
        )

        ro_pulses[pair[1]] = platform.create_MZ_pulse(pair[1], start=0)

        sequence.add(qd_pulses[pair[1]])
        sequence.add(cd_pulses[pair[0]])
        sequence.add(ro_pulses[pair[1]])

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(
        -params.frequency_width // 2, params.frequency_width // 2, params.frequency_step
    )

    sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[cd_pulses[pair[0]] for pair in qubit_pairs],
    )

    # create a DataUnits object to store the results,
    sweep_data = CouplerFrequencyData()

    # repeat the experiment as many times as defined by nshots
    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        sweeper,
    )

    # retrieve the results for every qubit
    for pair in qubit_pairs:
        # average msr, phase, i and q over the number of shots defined in the runcard
        qubit = pair[1]
        ro_pulse = ro_pulses[qubit]
        result = results[ro_pulse.serial]

        prob = np.abs(
            result.voltage_i
            + 1j * result.voltage_q
            - complex(platform.qubits[qubit].mean_gnd_states)
        ) / np.abs(
            complex(platform.qubits[qubit].mean_exc_states)
            - complex(platform.qubits[qubit].mean_gnd_states)
        )
        # prob = iq_to_probability(result.voltage_i, result.voltage_q, complex(platform.qubits[qubit].mean_exc_states), complex(platform.qubits[qubit].mean_gnd_states))
        # store the results
        r = {
            "frequency[Hz]": delta_frequency_range + params.coupler_frequency,
            "coupler": len(delta_frequency_range) * [f"c{pair[0]}"],
            "qubit": len(delta_frequency_range) * [qubit],
            "probability": prob,
        }
        sweep_data.add_data_from_dict(r)
    return sweep_data


def _plot(data: CouplerFrequencyData, fit: CouplerFrequencyResults, qubit):
    fig = go.Figure()

    # Plot data
    fig.add_trace(
        go.Scatter(
            x=data.df[(data.df["qubit"] == 2) & (data.df["coupler"] == f"c{qubit}")][
                "frequency"
            ]
            .pint.to("Hz")
            .pint.magnitude,
            y=data.df[(data.df["qubit"] == 2) & (data.df["coupler"] == f"c{qubit}")][
                "probability"
            ],
            mode="lines",
            name=f"Qubit {qubit}",
        )
    )
    fig.update_layout(
        title=f"Coupler frequency",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Probability",
        legend_title="Qubit",
    )
    return [fig], "No fitting data."


def _fit(data: CouplerFrequencyData):
    return CouplerFrequencyResults()


coupler_frequency = Routine(_aquisition, _fit, _plot)
"""Coupler frequency routine."""
