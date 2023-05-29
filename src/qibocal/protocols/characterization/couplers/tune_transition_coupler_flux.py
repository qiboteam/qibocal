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
class TuneTransitionCouplerFluxParameters(Parameters):
    """TuneTransitionCouplerFlux runcard inputs."""

    amplitude_factor_min: float
    """Amplitude minimum."""
    amplitude_factor_max: float
    """Amplitude maximum."""
    amplitude_factor_step: float
    """Amplitude step."""
    phase_min: float
    """Phase minimum."""
    phase_max: float
    """Phase maximum."""
    phase_step: float
    """Phase step."""
    nshots: Optional[int] = None
    """Number of shots per point."""
    wait_time: Optional[float] = 0
    """Wait time."""


@dataclass
class TuneTransitionCouplerFluxResults(Results):
    """TuneTransitionCouplerFlux outputs when fitting will be done."""


class TuneTransitionCouplerFluxData(DataUnits):
    """TuneTransitionCouplerFlux acquisition outputs."""

    def __init__(self):
        super().__init__(
            name="data",
            quantities={"amplitude": "dimensionless", "relative_phase": "rad"},
            options=[
                "coupler",
                "qubit",
                "probability",
                "state",
            ],
        )


def _aquisition(
    params: TuneTransitionCouplerFluxParameters,
    platform: AbstractPlatform,
    qubits: Qubits,
) -> TuneTransitionCouplerFluxData:
    r"""


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
    qd_pulses1 = {}
    qd_pulses2 = {}
    fx_pulses = {}

    for pair in qubit_pairs:
        qd_pulses1[pair[0]] = platform.create_RX90_pulse(pair[0], start=0)
        qd_pulses1[pair[1]] = platform.create_RX_pulse(
            pair[1], start=qd_pulses1[pair[0]].se_finish
        )
        qd_pulses2[pair[1]] = platform.create_RX_pulse(
            pair[1], start=qd_pulses1[pair[1]].se_finish + params.wait_time
        )
        qd_pulses2[pair[0]] = platform.create_RX90_pulse(
            pair[0], start=qd_pulses2[pair[1]].se_finish
        )

        ro_pulses[pair[1]] = platform.create_MZ_pulse(
            pair[1], start=qd_pulses2[pair[0]].se_finish
        )

        fx_pulses[pair[0]] = FluxPulse(
            start=0,
            duration=ro_pulses[pair[1]].se_start,
            amplitude=1,
            shape=Rectangular(),
            channel=platform.qubits[f"c{pair[0]}"].flux.name,
            qubit=f"c{pair[0]}",
        )

        sequence.add(qd_pulses1[pair[0]])
        sequence.add(qd_pulses1[pair[1]])
        sequence.add(qd_pulses2[pair[1]])
        sequence.add(qd_pulses2[pair[0]])
        sequence.add(fx_pulses[pair[0]])
        sequence.add(ro_pulses[pair[1]])

    # define the parameter to sweep and its range:
    delta_amplitude_range = np.arange(
        params.amplitude_factor_min,
        params.amplitude_factor_max,
        params.amplitude_factor_step,
    )
    delta_phase_range = np.arange(params.phase_min, params.phase_max, params.phase_step)

    sweeper_amplitude = Sweeper(
        Parameter.amplitude,
        delta_amplitude_range,
        pulses=[fx_pulses[pair[0]] for pair in qubit_pairs],
    )
    sweeper_phase = Sweeper(
        Parameter.relative_phase,
        delta_phase_range,
        pulses=[qd_pulses2[pair[0]] for pair in qubit_pairs],
    )

    # create a DataUnits object to store the results,
    sweep_data = TuneTransitionCouplerFluxData()

    # repeat the experiment as many times as defined by nshots
    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        sweeper_amplitude,
        sweeper_phase,
    )

    # retrieve the results for every qubit
    for pair in qubit_pairs:
        # WHEN MULTIPLEXING IS WORKING
        # for state, qubit in zip(["low","high"], pair):
        for state, qubit in zip(["high"], [pair[1]]):
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
            amp, phase = np.meshgrid(
                delta_amplitude_range, delta_phase_range, indexing="ij"
            )
            r = result.serialize
            # store the results
            r.update(
                {
                    "amplitude[dimensionless]": amp.flatten(),
                    "relative_phase[rad]": phase.flatten(),
                    "coupler": len(delta_amplitude_range)
                    * len(delta_phase_range)
                    * [f"c{pair[0]}"],
                    "qubit": len(delta_amplitude_range)
                    * len(delta_phase_range)
                    * [qubit],
                    "state": len(delta_amplitude_range)
                    * len(delta_phase_range)
                    * [state],
                    "probability": prob.flatten(),
                }
            )
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
        sweeper_phase,
    )

    for pair in qubit_pairs:
        # WHEN MULTIPLEXING IS WORKING
        # for state, qubit in zip(["low","high"], pair):
        for state, qubit in zip(["low"], [pair[0]]):
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
            r = result.serialize
            # store the results
            r.update(
                {
                    "amplitude[dimensionless]": amp.flatten(),
                    "relative_phase[rad]": phase.flatten(),
                    "coupler": len(delta_amplitude_range)
                    * len(delta_phase_range)
                    * [f"c{pair[0]}"],
                    "qubit": len(delta_amplitude_range)
                    * len(delta_phase_range)
                    * [qubit],
                    "state": len(delta_amplitude_range)
                    * len(delta_phase_range)
                    * [state],
                    "probability": prob.flatten(),
                }
            )
            sweep_data.add_data_from_dict(r)

    return sweep_data


def _plot(
    data: TuneTransitionCouplerFluxData, fit: TuneTransitionCouplerFluxResults, qubit
):
    states = ["low", "high"]

    figures = []
    colouraxis = ["coloraxis", "coloraxis2"]

    for values, unit in zip(["MSR", "phase", "probability"], ["V", "rad", None]):
        fig = make_subplots(rows=1, cols=2, subplot_titles=tuple(states))

        # Plot data
        for state, q in zip(
            states, [qubit, 2]
        ):  # When multiplex works zip(["low","high"], [qubit, 2])
            if unit is None:
                z = data.df[
                    (data.df["state"] == state)
                    & (data.df["qubit"] == q)
                    & (data.df["coupler"] == f"c{qubit}")
                ][values]
            else:
                z = data.df[
                    (data.df["state"] == state)
                    & (data.df["qubit"] == q)
                    & (data.df["coupler"] == f"c{qubit}")
                ][values]
                z = z.pint.to(unit).pint.magnitude

            fig.add_trace(
                go.Heatmap(
                    x=data.df[
                        (data.df["state"] == state)
                        & (data.df["qubit"] == q)
                        & (data.df["coupler"] == f"c{qubit}")
                    ]["relative_phase"]
                    .pint.to("rad")
                    .pint.magnitude,
                    y=data.df[
                        (data.df["state"] == state)
                        & (data.df["qubit"] == q)
                        & (data.df["coupler"] == f"c{qubit}")
                    ]["amplitude"]
                    .pint.to("dimensionless")
                    .pint.magnitude,
                    z=z,
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
            title=f"Qubit {qubit} cz {values}",
            xaxis_title="Duration [ns]",
            yaxis_title="Amplitude [dimensionless]",
            legend_title="States",
        )

        figures.append(fig)
    return figures, "No fitting data."


def _fit(data: TuneTransitionCouplerFluxData):
    return TuneTransitionCouplerFluxResults()


tune_transition_coupler_flux = Routine(_aquisition, _fit, _plot)
"""Coupler swap flux routine."""
