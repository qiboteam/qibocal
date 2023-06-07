from dataclasses import dataclass
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform, QubitPair
from qibolab.pulses import FluxPulse, Rectangular
from qibolab.sweeper import Parameter, Sweeper

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import DataUnits


@dataclass
class CouplerCzFluxTimeParameters(Parameters):
    """CouplerCzFluxTime runcard inputs."""

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
    dt: Optional[float] = 0
    """Wait time around the flux pulse in ns."""
    nshots: Optional[int] = None
    """Number of shots per point."""


@dataclass
class CouplerCzFluxTimeResults(Results):
    """CouplerCzFluxTime outputs when fitting will be done."""


class CouplerCzFluxTimeData(DataUnits):
    """CouplerCzFluxTime acquisition outputs."""

    def __init__(self):
        super().__init__(
            name="data",
            quantities={"amplitude": "dimensionless", "duration": "ns"},
            options=[
                "coupler",
                "qubit",
                "iq_distance",
                "state",
            ],
        )


def _aquisition(
    params: CouplerCzFluxTimeParameters,
    platform: Platform,
    qubits: Qubits,
) -> CouplerCzFluxTimeData:
    r"""
    Place the two qubits in the |11> state which oscillates to |02> by applying a flux pulse on the coupler.

    The qubits must be at specific frequencies such that the high frequency qubit
    1 to 2 transition is at the same frequency as the low frequency qubit 0 to 1 transition.
    At this avoided crossing, the coupling can be turned on and off by applying a flux pulse on the coupler.
    The amplitude of this flux pluse changes the frequency of the coupler. The
    closer the coupler frequency is to the avoided crossing, the stronger the coupling.
    A strong interaction allows for a faster CZ gate.

    Args:
        platform: Platform to use.
        params: Experiment parameters.
        qubits: Dict of QubitPairs.

    Returns:
        DataUnits: Acquisition data.
    """
    # define the parameter to sweep and its range:
    delta_amplitude_range = np.arange(
        params.amplitude_factor_min,
        params.amplitude_factor_max,
        params.amplitude_factor_step,
    )
    delta_duration_range = np.arange(
        params.duration_min, params.duration_max, params.duration_step
    )

    dur, amp = np.meshgrid(delta_duration_range, delta_amplitude_range, indexing="ij")

    # create a DataUnits object to store the results,
    sweep_data = CouplerCzFluxTimeData()

    # sort high and low frequency qubit
    for pair in qubits:
        pair = sort_frequency(pair)

        q_lowfreq = pair.qubit1.name
        q_highfreq = pair.qubit2.name

        qd_pulse1 = platform.create_RX_pulse(q_lowfreq, start=0)
        qd_pulse2 = platform.create_RX_pulse(q_highfreq, start=0 + params.dt)
        sequence = qd_pulse1 + qd_pulse2
        fx_pulse = FluxPulse(
            start=sequence.finish + params.dt,
            duration=params.duration_min,
            amplitude=1,
            shape=Rectangular(),
            channel=pair[1].coupler.flux.name,
            qubit=pair[1].coupler.name,
        )
        sequence += fx_pulse

        ro_pulse1 = platform.create_MZ_pulse(
            q_highfreq, start=sequence.finish + params.dt
        )
        ro_pulse2 = platform.create_MZ_pulse(
            q_lowfreq, start=sequence.finish + params.dt
        )

        sequence += ro_pulse1 + ro_pulse2

        sweeper_amplitude = Sweeper(
            Parameter.amplitude,
            delta_amplitude_range,
            pulses=[fx_pulse],
        )
        sweeper_duration = Sweeper(
            Parameter.duration,
            delta_duration_range,
            pulses=[fx_pulse],
        )

        # repeat the experiment as many times as defined by nshots
        results = platform.sweep(
            sequence,
            ExecutionParameters(
                nshots=params.nshots,
                acquisition_type=AcquisitionType.INTEGRATION,
                averaging_mode=AveragingMode.CYCLIC,
            ),
            sweeper_duration,
            sweeper_amplitude,
        )

        for state, ro_pulse in zip(["low", "high"], [ro_pulse2, ro_pulse1]):
            # average msr, phase, i and q over the number of shots defined in the runcard
            result = results[ro_pulse.serial]

            iq_distance = np.abs(
                result.voltage_i
                + 1j * result.voltage_q
                - complex(platform.qubits[ro_pulse.qubit].mean_gnd_states)
            ) / np.abs(
                complex(platform.qubits[ro_pulse.qubit].mean_exc_states)
                - complex(platform.qubits[ro_pulse.qubit].mean_gnd_states)
            )

            r = result.serialize
            # store the results
            r.update(
                {
                    "amplitude[dimensionless]": amp.flatten(),
                    "duration[ns]": dur.flatten(),
                    "coupler": len(delta_amplitude_range)
                    * len(delta_duration_range)
                    * [f"c{q_lowfreq}"],
                    "qubit": len(delta_amplitude_range)
                    * len(delta_duration_range)
                    * [ro_pulse.qubit],
                    "state": len(delta_amplitude_range)
                    * len(delta_duration_range)
                    * [state],
                    "iq_distance": iq_distance.flatten(),
                }
            )
            sweep_data.add_data_from_dict(r)

    return sweep_data


def _plot(data: CouplerCzFluxTimeData, fit: CouplerCzFluxTimeResults, pair: QubitPair):
    states = ["low", "high"]

    figures = []
    colouraxis = ["coloraxis", "coloraxis2"]
    pair = sort_frequency(pair)

    for values, unit in zip(["MSR", "phase", "iq_distance"], ["V", "rad", None]):
        fig = make_subplots(rows=1, cols=2, subplot_titles=tuple(states))

        # Plot data
        for state, q in zip(
            states, pair
        ):  # When multiplex works zip(["low","high"], [qubit, 2])
            if unit is None:
                z = data.df[
                    (data.df["state"] == state)
                    & (data.df["qubit"] == q)
                    & (data.df["coupler"] == pair[1].coupler.name)
                ][values]
            else:
                z = data.df[
                    (data.df["state"] == state)
                    & (data.df["qubit"] == q)
                    & (data.df["coupler"] == pair[1].coupler.name)
                ][values]
                z = z.pint.to(unit).pint.magnitude

            fig.add_trace(
                go.Heatmap(
                    y=data.df[
                        (data.df["state"] == state)
                        & (data.df["qubit"] == q)
                        & (data.df["coupler"] == pair[1].coupler.name)
                    ]["duration"]
                    .pint.to("ns")
                    .pint.magnitude,
                    x=data.df[
                        (data.df["state"] == state)
                        & (data.df["qubit"] == q)
                        & (data.df["coupler"] == pair[1].coupler.name)
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
            title=f"Qubit {pair.name} cz {values}",
            yaxis_title="Duration [ns]",
            xaxis_title="Amplitude [dimensionless]",
            legend_title="States",
        )

        figures.append(fig)
    return figures, "No fitting data."


def _fit(data: CouplerCzFluxTimeData):
    return CouplerCzFluxTimeResults()


def sort_frequency(pair: QubitPair):
    """Sorts the qubits in a pair by frequency."""
    if pair.qubit1.frequency > pair.qubit2.frequency:
        return pair.qubit2, pair.qubit1
    else:
        return pair.qubit1, pair.qubit2


cz_flux_time = Routine(_aquisition, _fit, _plot)
"""Coupler swap flux routine."""
