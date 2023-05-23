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
from qibolab.pulses import FluxPulse, PulseSequence, Rectangular
from qibolab.sweeper import Parameter, Sweeper

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import DataUnits


@dataclass
class CouplerSwapFluxParameters(Parameters):
    """CouplerSwapFlux runcard inputs."""

    amplitude_factor_min: float
    """Amplitude minimum."""
    amplitude_factor_max: float
    """Amplitude maximum."""
    amplitude_factor_step: float
    """Amplitude step."""
    nshots: Optional[int] = None
    """Number of shots per point."""
    coupler_flux_duration: Optional[float] = None
    """Coupler flux duration."""


@dataclass
class CouplerSwapFluxResults(Results):
    """CouplerSwapFlux outputs when fitting will be done."""


class CouplerSwapFluxData(DataUnits):
    """CouplerSwapFlux acquisition outputs."""

    def __init__(self):
        super().__init__(
            name="data",
            quantities={"amplitude": "dimensionless"},
            options=[
                "coupler",
                "qubit",
                "probability",
                "state",
            ],
        )


def _aquisition(
    params: CouplerSwapFluxParameters,
    platform: AbstractPlatform,
    qubits: Qubits,
) -> CouplerSwapFluxData:
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
    qubit_pairs = list([qubit, 2] for qubit in qubits)

    # create a sequence
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    fx_pulses = {}

    # FIXME: add coupler pulses in the runcard
    if params.coupler_flux_duration is None:
        params.coupler_flux_duration = 2000
    for pair in qubit_pairs:
        qd_pulses[pair[1]] = platform.create_RX_pulse(pair[1], start=0)
        fx_pulses[pair[0]] = FluxPulse(
            start=qd_pulses[pair[1]].se_finish + 8,
            duration=params.coupler_flux_duration,
            amplitude=1,
            shape=Rectangular(),
            channel=platform.qubits[f"c{pair[0]}"].flux.name,
        )

        ro_pulses[pair[1]] = platform.create_MZ_pulse(
            pair[1], start=fx_pulses[pair[0]].se_finish + 8
        )
        # ro_pulses[pair[0]] = platform.create_MZ_pulse(pair[0], start=ro_pulses[pair[1]].se_finish) # Multiplex not working yet

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

    sweeper = Sweeper(
        Parameter.amplitude,
        delta_amplitude_range,
        pulses=[fx_pulses[pair[0]] for pair in qubit_pairs],
    )

    # create a DataUnits object to store the results,
    sweep_data = CouplerSwapFluxData()

    # repeat the experiment as many times as defined by nshots
    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        sweeper,
    )

    # retrieve the results for every qubit
    for pair in qubit_pairs:
        # WHEN MULTIPLEXING IS WORKING
        # for state, qubit in zip([0, 1], pair):
        for state, qubit in zip([1], [pair[1]]):
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
            # prob = iq_to_probability(result.voltage_i, result.voltage_q, complex(platform.qubits[qubit].mean_exc_states), complex(platform.qubits[qubit].mean_gnd_states))
            # store the results
            r = {
                "amplitude[dimensionless]": delta_amplitude_range,
                "coupler": len(delta_amplitude_range) * [f"c{pair[0]}"],
                "qubit": len(delta_amplitude_range) * [qubit],
                "state": len(delta_amplitude_range) * [state],
                "probability": prob,
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
        sweeper,
    )

    for pair in qubit_pairs:
        # WHEN MULTIPLEXING IS WORKING
        # for state, qubit in zip([0, 1], pair):
        for state, qubit in zip([0], [pair[0]]):
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
                "amplitude[dimensionless]": delta_amplitude_range,
                "coupler": len(delta_amplitude_range) * [f"c{pair[0]}"],
                "qubit": len(delta_amplitude_range) * [qubit],
                "state": len(delta_amplitude_range) * [state],
                "probability": prob,
            }
            sweep_data.add_data_from_dict(r)

    return sweep_data


def _plot(data: CouplerSwapFluxData, fit: CouplerSwapFluxResults, qubit):
    fig = go.Figure()

    # Plot data
    for state, q in zip(
        [0, 1], [qubit, 2]
    ):  # When multiplex works zip([0, 1], [qubit, 2])
        fig.add_trace(
            go.Scatter(
                x=data.df[
                    (data.df["state"] == state)
                    & (data.df["qubit"] == q)
                    & (data.df["coupler"] == f"c{qubit}")
                ]["amplitude"]
                .pint.to("dimensionless")
                .pint.magnitude,
                y=data.df[
                    (data.df["state"] == state)
                    & (data.df["qubit"] == q)
                    & (data.df["coupler"] == f"c{qubit}")
                ]["probability"],
                mode="lines",
                name=f"Qubit {q} |{state}>",
            )
        )
    fig.update_layout(
        title=f"Qubit {qubit} swap flux",
        xaxis_title="Amplitude [dimensionless]",
        yaxis_title="Probability",
        legend_title="States",
    )

    return [fig], "No fitting data."


def _fit(data: CouplerSwapFluxData):
    return CouplerSwapFluxResults()


coupler_swap_flux = Routine(_aquisition, _fit, _plot)
"""Coupler swap flux routine."""
