from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import DataUnits


@dataclass
class QubitFluxParameters(Parameters):
    """QubitFlux runcard inputs."""

    offset_width: float
    """Width of the offset range."""
    offset_step: float
    """Step of the offset range."""
    nshots: Optional[int] = 1024
    """Number of shots per point."""
    relaxation_time: Optional[float] = 0
    """Relaxation time."""


@dataclass
class QubitFluxResults(Results):
    """QubitFlux outputs when fitting will be done."""


class QubitFluxData(DataUnits):
    """QubitFlux acquisition outputs."""

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
    params: QubitFluxParameters,
    platform: AbstractPlatform,
    qubits: Qubits,
) -> QubitFluxData:
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
        if len(qubits) == 1:
            raise ValueError(
                "Please specify wich coupler needs to be used with qubit 2."
            )
        qubits.pop(2)

    # create a sequence
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}

    for qubit in qubits:
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=0, duration=2000
        )
        ro_pulses[qubit] = platform.create_MZ_pulse(qubit, start=0)
        ro_pulses[qubit].duration = qd_pulses[qubit].duration

        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    delta_offset_range = np.arange(
        -params.offset_width / 2, params.offset_width / 2, params.offset_step
    )

    sweeper_offset = Sweeper(
        Parameter.bias,
        delta_offset_range,
        qubits=[platform.qubits[f"c{qubit}"] for qubit in qubits],
    )

    # create a DataUnits object to store the results,
    sweep_data = QubitFluxData()

    # repeat the experiment as many times as defined by nshots
    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        sweeper_offset,
    )

    # retrieve the results for every qubit
    for qubit in qubits:
        # average msr, phase, i and q over the number of shots defined in the runcard
        ro_pulse = ro_pulses[qubit]
        result = results[ro_pulse.serial]

        r = {k: v.ravel() for k, v in result.serialize.items()}

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

        r.update(
            {
                "offset[V]": delta_offset_range
                + platform.qubits[f"c{qubit}"].sweetspot,
                "coupler": len(delta_offset_range) * [f"c{qubit}"],
                "qubit": len(delta_offset_range) * [qubit],
                "probability": prob.flatten(),
            }
        )
        sweep_data.add_data_from_dict(r)
    return sweep_data


def _plot(data: QubitFluxData, fit: QubitFluxResults, qubit):
    figs = []

    fig = go.Figure()

    # Plot data
    fig.add_trace(
        go.Scatter(
            x=data.df[
                (data.df["qubit"] == qubit) & (data.df["coupler"] == f"c{qubit}")
            ]["offset"]
            .pint.to("V")
            .pint.magnitude,
            y=data.df[
                (data.df["qubit"] == qubit) & (data.df["coupler"] == f"c{qubit}")
            ]["probability"],
            name=f"Qubit {qubit}",
        )
    )
    fig.update_layout(
        title=f"Resonator {qubit} flux",
        xaxis_title="Offset (V)",
        yaxis_title="Probability",
    )

    fig_2 = make_subplots(rows=1, cols=2, subplot_titles=("MSR", "Phase"))
    # Plot MSR and phase
    fig_2.add_trace(
        go.Scatter(
            x=data.df[
                (data.df["qubit"] == qubit) & (data.df["coupler"] == f"c{qubit}")
            ]["offset"]
            .pint.to("V")
            .pint.magnitude,
            y=data.df[
                (data.df["qubit"] == qubit) & (data.df["coupler"] == f"c{qubit}")
            ]["MSR"]
            .pint.to("V")
            .pint.magnitude,
            name=f"Qubit {qubit}",
        ),
        row=1,
        col=1,
    )
    fig_2.add_trace(
        go.Scatter(
            x=data.df[
                (data.df["qubit"] == qubit) & (data.df["coupler"] == f"c{qubit}")
            ]["offset"]
            .pint.to("V")
            .pint.magnitude,
            y=data.df[
                (data.df["qubit"] == qubit) & (data.df["coupler"] == f"c{qubit}")
            ]["phase"]
            .pint.to("rad")
            .pint.magnitude,
            name=f"Qubit {qubit}",
        ),
        row=1,
        col=2,
    )

    fig_2.update_layout(
        title=f"Resonator {qubit} flux map",
        xaxis_title="Offset [V]",
        yaxis_title="phase [rad]",
    )

    figs.append(fig)
    figs.append(fig_2)
    return figs, "No fitting data."


def _fit(data: QubitFluxData):
    return QubitFluxResults()


qubit_flux = Routine(_aquisition, _fit, _plot)
"""Coupler frequency flux routine."""
