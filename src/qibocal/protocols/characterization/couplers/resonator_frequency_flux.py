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
class ResonatorFrequencyFluxParameters(Parameters):
    """ResonatorFrequencyFlux runcard inputs."""

    frequency_width: float
    """Frequency width."""
    frequency_step: float
    """Frequency step."""
    offset_width: float
    """Flux offset width in volt."""
    offset_step: float
    """Flux offset step in volt."""
    nshots: Optional[int] = 1024
    """Number of shots per point."""
    relaxation_time: Optional[float] = 0
    """Relaxation time."""


@dataclass
class ResonatorFrequencyFluxResults(Results):
    """ResonatorFrequencyFlux outputs when fitting will be done."""


class ResonatorFrequencyFluxData(DataUnits):
    """ResonatorFrequencyFlux acquisition outputs."""

    def __init__(self):
        super().__init__(
            name="data",
            quantities={"frequency": "Hz", "offset": "V"},
            options=[
                "coupler",
                "qubit",
                "probability",
            ],
        )


def _aquisition(
    params: ResonatorFrequencyFluxParameters,
    platform: AbstractPlatform,
    qubits: Qubits,
) -> ResonatorFrequencyFluxData:
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

    for qubit in qubits:
        ro_pulses[qubit] = platform.create_MZ_pulse(qubit, start=0)

        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(
        -params.frequency_width // 2, params.frequency_width // 2, params.frequency_step
    )
    delta_offset_range = np.arange(
        -params.offset_width / 2, params.offset_width / 2, params.offset_step
    )

    sweeper_frequency = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[ro_pulses[qubit] for qubit in qubits],
    )

    sweeper_offset = Sweeper(
        Parameter.bias,
        delta_offset_range,
        qubits=[platform.qubits[f"c{qubit}"] for qubit in qubits],
    )

    # create a DataUnits object to store the results,
    sweep_data = ResonatorFrequencyFluxData()

    # repeat the experiment as many times as defined by nshots
    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        sweeper_frequency,
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
        freq, offset = np.meshgrid(
            delta_frequency_range, delta_offset_range, indexing="ij"
        )

        r.update(
            {
                "frequency[Hz]": freq.flatten() + ro_pulses[qubit].frequency,
                "offset[V]": offset.flatten() + platform.qubits[f"c{qubit}"].sweetspot,
                "coupler": len(delta_offset_range)
                * len(delta_frequency_range)
                * [f"c{qubit}"],
                "qubit": len(delta_frequency_range) * len(delta_offset_range) * [qubit],
                "probability": prob.flatten(),
            }
        )
        sweep_data.add_data_from_dict(r)
    return sweep_data


def _plot(data: ResonatorFrequencyFluxData, fit: ResonatorFrequencyFluxResults, qubit):
    figs = []

    fig = go.Figure()

    # Plot data
    fig.add_trace(
        go.Heatmap(
            x=data.df[
                (data.df["qubit"] == qubit) & (data.df["coupler"] == f"c{qubit}")
            ]["frequency"]
            .pint.to("Hz")
            .pint.magnitude,
            y=data.df[
                (data.df["qubit"] == qubit) & (data.df["coupler"] == f"c{qubit}")
            ]["offset"]
            .pint.to("V")
            .pint.magnitude,
            z=data.df[
                (data.df["qubit"] == qubit) & (data.df["coupler"] == f"c{qubit}")
            ]["probability"],
            name=f"Qubit {qubit}",
        )
    )
    fig.update_layout(
        title=f"Resonator {qubit} flux map",
        xaxis_title="Frequency [Hz]",
        yaxis_title="Offset [V]",
    )

    fig_2 = make_subplots(rows=1, cols=2, subplot_titles=("MSR", "Phase"))
    # Plot MSR and phase
    fig_2.add_trace(
        go.Heatmap(
            x=data.df[
                (data.df["qubit"] == qubit) & (data.df["coupler"] == f"c{qubit}")
            ]["frequency"]
            .pint.to("Hz")
            .pint.magnitude,
            y=data.df[
                (data.df["qubit"] == qubit) & (data.df["coupler"] == f"c{qubit}")
            ]["offset"]
            .pint.to("V")
            .pint.magnitude,
            z=data.df[
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
        go.Heatmap(
            x=data.df[
                (data.df["qubit"] == qubit) & (data.df["coupler"] == f"c{qubit}")
            ]["frequency"]
            .pint.to("Hz")
            .pint.magnitude,
            y=data.df[
                (data.df["qubit"] == qubit) & (data.df["coupler"] == f"c{qubit}")
            ]["offset"]
            .pint.to("V")
            .pint.magnitude,
            z=data.df[
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
        xaxis_title="Frequency [Hz]",
        yaxis_title="Offset [V]",
    )

    figs.append(fig)
    figs.append(fig_2)
    return figs, "No fitting data."


def _fit(data: ResonatorFrequencyFluxData):
    return ResonatorFrequencyFluxResults()


resonator_frequency_flux = Routine(_aquisition, _fit, _plot)
"""Coupler frequency flux routine."""
