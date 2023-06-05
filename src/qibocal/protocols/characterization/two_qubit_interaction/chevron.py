from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import FluxPulse, PulseSequence, Rectangular
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import DataUnits


@dataclass
class ChevronParameters(Parameters):
    """CzFluxTime runcard inputs."""

    amplitude_factor_min: float
    """Amplitude minimum."""
    amplitude_factor_max: float
    """Amplitude maximum."""
    amplitude_factor_step: float
    """Amplitude step."""
    duration_min: float
    """Duration minimum."""
    duration_max: float
    """Duration maximum."""
    duration_step: float
    """Duration step."""
    qubits: Optional[list[list[QubitId, QubitId]]] = None
    """Pair(s) of qubit to probe."""
    nshots: Optional[int] = None
    """Number of shots per point."""


@dataclass
class ChevronResults(Results):
    """CzFluxTime outputs when fitting will be done."""


class ChevronData(DataUnits):
    """CzFluxTime acquisition outputs."""

    def __init__(self):
        super().__init__(
            name="data",
            quantities={"amplitude": "dimensionless", "duration": "ns"},
            options=[
                "qubit",
                "probability",
                "state",
            ],
        )


def _aquisition(
    params: ChevronParameters,
    platform: Platform,
    qubits: Qubits,  # TODO this parameter is not used so probably I did it wrong
) -> ChevronData:
    r"""
    Perform a SWAP experiment between pairs of qubits by changing its frequency.

    Args:
        platform: Platform to use.
        params: Experiment parameters.
        qubits: Qubits to use.

    Returns:
        DataUnits: Acquisition data.
    """
    if params.qubits is None:
        params.qubits = platform.settings["topology"]

    # create a DataUnits object to store the results,
    sweep_data = ChevronData()
    for pair in params.qubits:
        # order the qubits so that the low frequency one is the first
        if (
            platform.qubits[pair[0]].drive_frequency
            > platform.qubits[pair[1]].drive_frequency
        ):
            ord_pair = pair[::-1]
        else:
            ord_pair = pair

        # create a sequence
        sequence = PulseSequence()
        ro_pulses = {}
        qd_pulses = {}
        fx_pulses = {}

        qd_pulses[ord_pair[0]] = platform.create_RX_pulse(ord_pair[0], start=0)
        qd_pulses[ord_pair[1]] = platform.create_RX_pulse(ord_pair[1], start=0)
        fx_pulses[ord_pair[0]] = FluxPulse(
            start=max(
                [qd_pulses[ord_pair[0]].se_finish, qd_pulses[ord_pair[1]].se_finish]
            )
            + 8,
            duration=params.duration_min,
            amplitude=1,
            shape=Rectangular(),
            channel=platform.qubits[ord_pair[0]].flux.name,
            qubit=ord_pair[0],
        )

        ro_pulses[ord_pair[0]] = platform.create_MZ_pulse(
            ord_pair[0], start=fx_pulses[ord_pair[0]].se_finish + 8
        )
        ro_pulses[ord_pair[1]] = platform.create_MZ_pulse(
            ord_pair[1], start=fx_pulses[ord_pair[0]].se_finish + 8
        )

        sequence.add(qd_pulses[ord_pair[0]])
        sequence.add(qd_pulses[ord_pair[1]])
        sequence.add(fx_pulses[ord_pair[0]])
        sequence.add(ro_pulses[ord_pair[0]])
        sequence.add(ro_pulses[ord_pair[1]])

        # define the parameter to sweep and its range:
        delta_amplitude_range = np.arange(
            params.amplitude_factor_min,
            params.amplitude_factor_max,
            params.amplitude_factor_step,
        )
        delta_duration_range = np.arange(
            params.duration_min, params.duration_max, params.duration_step
        )

        sweeper_amplitude = Sweeper(
            Parameter.amplitude,
            delta_amplitude_range,
            pulses=[fx_pulses[ord_pair[0]]],
        )
        sweeper_duration = Sweeper(
            Parameter.duration,
            delta_duration_range,
            pulses=[fx_pulses[ord_pair[0]]],
        )

        results = platform.sweep(
            sequence,
            ExecutionParameters(
                nshots=params.nshots,
                acquisition_type=AcquisitionType.DISCRIMINATION,
                averaging_mode=AveragingMode.CYCLIC,
            ),
            sweeper_amplitude,
            sweeper_duration,
        )

        # TODO: here it is written ["high", "low"] but the low frequency is pair[0]
        #       was it correct?
        for state, qubit in zip(["high", "low"], pair):
            # average msr, phase, i and q over the number of shots defined in the runcard
            ro_pulse = ro_pulses[qubit]

            result = results[ro_pulse.serial]
            prob = result.statistical_frequency
            amp, dur = np.meshgrid(
                delta_amplitude_range, delta_duration_range, indexing="ij"
            )
            # store the results
            r = {
                "amplitude[dimensionless]": amp.flatten(),
                "duration[ns]": dur.flatten(),
                "qubit": (
                    len(delta_amplitude_range)
                    * len(delta_duration_range)
                    * [f"{qubit}_{pair}"]
                ),
                "state": (
                    len(delta_amplitude_range)
                    * len(delta_duration_range)
                    * [f"{state}_{pair}"]
                ),
                "probability": prob.flatten(),
            }
            sweep_data.add_data_from_dict(r)

    return sweep_data


def _plot(data: ChevronData, fit: ChevronResults, qubits):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("high", "low"))
    states = ["high", "low"]
    # Plot data
    colouraxis = ["coloraxis", "coloraxis2"]
    for state, q in zip(states, qubits):
        fig.add_trace(
            go.Heatmap(
                x=data.df[
                    (data.df["state"] == f"{state}_{qubits}")
                    & (data.df["qubit"] == f"{q}_{qubits}")
                ]["duration"]
                .pint.to("ns")
                .pint.magnitude,
                y=data.df[
                    (data.df["state"] == f"{state}_{qubits}")
                    & (data.df["qubit"] == f"{q}_{qubits}")
                ]["amplitude"]
                .pint.to("dimensionless")
                .pint.magnitude,
                z=data.df[
                    (data.df["state"] == f"{state}_{qubits}")
                    & (data.df["qubit"] == f"{q}_{qubits}")
                ]["probability"],
                name=f"Qubit {q} |{state}>",
                coloraxis=colouraxis[states.index(state)],
            ),
            row=1,
            col=states.index(state) + 1,
        )

    fig.update_layout(
        title=f"Qubits {qubits[0]}-{qubits[1]} swap frequency",
        xaxis_title="Duration [ns]",
        yaxis_title="Amplitude [dimensionless]",
        legend_title="States",
    )
    fig.update_layout(
        coloraxis=dict(colorscale="Oryel", colorbar=dict(x=-0.15)),
        coloraxis2=dict(colorscale="Darkmint", colorbar=dict(x=1.15)),
    )
    return [fig], "No fitting data."


def _fit(data: ChevronData):
    return ChevronResults()


chevron = Routine(_aquisition, _fit, _plot)
"""Chevron routine."""
