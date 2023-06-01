from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import FluxPulse, PulseSequence, Rectangular
from qibolab.sweeper import Parameter, Sweeper

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import DataUnits


@dataclass
class CzFluxTimeParameters(Parameters):
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
    nshots: Optional[int] = None
    """Number of shots per point."""


@dataclass
class CzFluxTimeResults(Results):
    """CzFluxTime outputs when fitting will be done."""


class CzFluxTimeData(DataUnits):
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
    params: CzFluxTimeParameters,
    platform: AbstractPlatform,
    qubits: Qubits,
) -> CzFluxTimeData:
    r"""
    Perform a SWAP experiment between two qubits through the coupler by changing its frequency.

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
    qubit_pairs = [
        [qubit, 2]
        if platform.qubits[qubit].drive_frequency > platform.qubits[2].drive_frequency
        else [2, qubit]
        for qubit in qubits
    ]

    # create a sequence
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    fx_pulses = {}

    for pair in qubit_pairs:
        qd_pulses[pair[0]] = platform.create_RX_pulse(pair[0], start=0)
        qd_pulses[pair[1]] = platform.create_RX_pulse(pair[1], start=0)
        fx_pulses[pair[0]] = FluxPulse(
            start=max([qd_pulses[pair[0]].se_finish, qd_pulses[pair[1]].se_finish]) + 8,
            duration=params.duration_min,
            amplitude=1,
            shape=Rectangular(),
            channel=platform.qubits[pair[0]].flux.name,
            qubit=pair[0],
        )

        ro_pulses[pair[1]] = platform.create_MZ_pulse(
            pair[1], start=fx_pulses[pair[0]].se_finish + 8
        )
        # ro_pulses[pair[0]] = platform.create_MZ_pulse(pair[0], start=ro_pulses[pair[1]].se_finish) # Multiplex not working yet

        sequence.add(qd_pulses[pair[0]])
        sequence.add(qd_pulses[pair[1]])
        sequence.add(fx_pulses[pair[0]])
        sequence.add(ro_pulses[pair[1]])
        # sequence.add(ro_pulses[pair[0]])

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
        pulses=[fx_pulses[pair[0]] for pair in qubit_pairs],
    )
    sweeper_duration = Sweeper(
        Parameter.duration,
        delta_duration_range,
        pulses=[fx_pulses[pair[0]] for pair in qubit_pairs],
    )

    # create a DataUnits object to store the results,
    sweep_data = CzFluxTimeData()

    # repeat the experiment as many times as defined by nshots
    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        sweeper_amplitude,
        sweeper_duration,
    )

    # retrieve the results for every qubit
    for pair in qubit_pairs:
        # WHEN MULTIPLEXING IS WORKING
        # for state, qubit in zip(["high", "low"], pair):
        for state, qubit in zip(["low"], [pair[1]]):
            # average msr, phase, i and q over the number of shots defined in the runcard
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
            amp, dur = np.meshgrid(
                delta_amplitude_range, delta_duration_range, indexing="ij"
            )
            # store the results
            r = {
                "amplitude[dimensionless]": amp.flatten(),
                "duration[ns]": dur.flatten(),
                "qubit": len(delta_amplitude_range)
                * len(delta_duration_range)
                * [qubit],
                "state": len(delta_amplitude_range)
                * len(delta_duration_range)
                * [state],
                "probability": prob.flatten(),
            }
            sweep_data.add_data_from_dict(r)

    # Temporary fix for multiplexing, repeat the experiment for the second qubit
    ro_pulses[pair[0]] = platform.create_MZ_pulse(
        pair[0], start=ro_pulses[pair[1]].se_start
    )  # Multiplex not working yet
    sequence.add(ro_pulses[pair[0]])
    sequence.remove(ro_pulses[pair[1]])

    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        sweeper_amplitude,
        sweeper_duration,
    )

    for pair in qubit_pairs:
        # WHEN MULTIPLEXING IS WORKING
        # for state, qubit in zip(["high", "low"], pair):
        for state, qubit in zip(["high"], [pair[0]]):
            # average msr, phase, i and q over the number of shots defined in the runcard
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
            # store the results
            r = {
                "amplitude[dimensionless]": amp.flatten(),
                "duration[ns]": dur.flatten(),
                "qubit": len(delta_amplitude_range)
                * len(delta_duration_range)
                * [qubit],
                "state": len(delta_amplitude_range)
                * len(delta_duration_range)
                * [state],
                "probability": prob.flatten(),
            }
            sweep_data.add_data_from_dict(r)

    return sweep_data


def _plot(data: CzFluxTimeData, fit: CzFluxTimeResults, qubit):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("high", "low"))
    states = ["high", "low"]
    # Plot data
    colouraxis = ["coloraxis", "coloraxis2"]
    for state, q in zip(
        states, [2, qubit]
    ):  # When multiplex works zip(["high", "low"], [qubit, 2])
        fig.add_trace(
            go.Heatmap(
                x=data.df[(data.df["state"] == state) & (data.df["qubit"] == q)][
                    "duration"
                ]
                .pint.to("ns")
                .pint.magnitude,
                y=data.df[(data.df["state"] == state) & (data.df["qubit"] == q)][
                    "amplitude"
                ]
                .pint.to("dimensionless")
                .pint.magnitude,
                z=data.df[(data.df["state"] == state) & (data.df["qubit"] == q)][
                    "probability"
                ],
                name=f"Qubit {q} |{state}>",
                coloraxis=colouraxis[states.index(state)],
            ),
            row=1,
            col=states.index(state) + 1,
        )

        fig.update_layout(
            coloraxis=dict(colorscale="Viridis", colorbar=dict(x=0.45)),
            coloraxis2=dict(colorscale="Cividis", colorbar=dict(x=1)),
        )
    fig.update_layout(
        title=f"Qubit {qubit} swap frequency",
        xaxis_title="Duration [ns]",
        yaxis_title="Amplitude [dimensionless]",
        legend_title="States",
    )
    fig.update_layout(
        coloraxis=dict(colorscale="Viridis", colorbar=dict(x=-0.15)),
        coloraxis2=dict(colorscale="Cividis", colorbar=dict(x=1.15)),
    )
    return [fig], "No fitting data."


def _fit(data: CzFluxTimeData):
    return CzFluxTimeResults()


cz_flux_time = Routine(_aquisition, _fit, _plot)
"""Coupler swap flux routine."""
