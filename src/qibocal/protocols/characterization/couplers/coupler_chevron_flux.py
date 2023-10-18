from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.qubits import QubitId, QubitPair
from qibolab.sweeper import Parameter, Sweeper

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine

from ..two_qubit_interaction.utils import order_pair


@dataclass
class ChevronFluxTimeParameters(Parameters):
    """ChevronFluxTime runcard inputs."""

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
    native_gate: Optional[str] = "CZ"
    """Native gate to implement, CZ or iSWAP."""
    dt: Optional[float] = 0
    """Wait time around the flux pulse in ns."""
    nshots: Optional[int] = None
    """Number of shots per point."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class ChevronFluxTimeResults(Results):
    """ChevronFluxTime outputs when fitting will be done."""


ChevronFluxTimeType = np.dtype(
    [
        ("amplitude", np.float64),
        ("duration", np.float64),
        ("msr", np.float64),
        ("phase", np.float64),
        ("iq_distance", np.float64),
    ]
)


@dataclass
class ChevronFluxTimeData(Data):
    """ChevronFluxTime acquisition outputs."""

    # qubit_pairs: dict[QubitId, QubitPair]
    data: dict[tuple[QubitId, int, int], npt.NDArray[ChevronFluxTimeType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""

    def register_qubit(
        self, qubit, coupler, state, amps, durs, msr, phase, iq_distance
    ):
        """Store output for single qubit."""
        size = len(amps) * len(durs)
        ar = np.empty(size, dtype=ChevronFluxTimeType)
        amplitude, durations = np.meshgrid(amps, durs)
        ar["amplitude"] = amplitude.ravel()
        ar["duration"] = durations.ravel()
        ar["msr"] = msr.ravel()
        ar["phase"] = phase.ravel()
        ar["iq_distance"] = iq_distance.ravel()
        self.data[qubit, coupler, state] = np.rec.array(ar)


def _aquisition(
    params: ChevronFluxTimeParameters,
    platform: Platform,
    qubits: Qubits,
) -> ChevronFluxTimeData:
    r"""
    Routine to find the optimal flux pulse amplitude and duration for a CZ/iSWAP gate.

    The qubits must be at specific frequencies such that the high frequency qubit
    1 to 2 (CZ) / 0 to 1 (iSWAP) transition is at the same frequency as the low frequency qubit 0 to 1 transition.
    At this avoided crossing, the coupling can be turned on and off by applying a flux pulse on the coupler.
    The amplitude of this flux pluse changes the frequency of the coupler. The
    closer the coupler frequency is to the avoided crossing, the stronger the coupling.
    A strong interaction allows for a faster controlled gate.

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

    # create a DataUnits object to store the results,
    data = ChevronFluxTimeData()

    # sort high and low frequency qubit
    for pair in qubits:
        # TODO: Qubit pair patch
        ordered_pair = order_pair(pair, platform.qubits)
        coupler = platform.pairs[tuple(sorted(ordered_pair))].coupler

        pair = ordered_pair
        # pair = sort_frequency(pair)
        # q_lowfreq = pair.qubit1.name
        # q_highfreq = pair.qubit2.name

        q_lowfreq = pair[0]
        q_highfreq = pair[1]

        qd_pulse1 = platform.create_RX_pulse(q_lowfreq, start=0)
        if params.native_gate == "iSWAP":
            qd_pulse1.amplitude = 0
        qd_pulse2 = platform.create_RX_pulse(q_highfreq, start=0 + params.dt)
        sequence = qd_pulse1 + qd_pulse2

        fx_pulse = platform.create_coupler_pulse(
            coupler=coupler,
            start=sequence.finish + params.dt,
            duration=params.duration_min,
            amplitude=1,
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
                - (
                    platform.qubits[ro_pulse.qubit].mean_gnd_states[0]
                    + platform.qubits[ro_pulse.qubit].mean_gnd_states[0] * 1j
                )
            ) / np.abs(
                platform.qubits[ro_pulse.qubit].mean_exc_states[0]
                + platform.qubits[ro_pulse.qubit].mean_exc_states[0] * 1j
                - platform.qubits[ro_pulse.qubit].mean_gnd_states[0]
                + platform.qubits[ro_pulse.qubit].mean_gnd_states[0] * 1j
            )

            data.register_qubit(
                qubit=ro_pulse.qubit,
                coupler=coupler.name,
                state=state,
                amps=delta_amplitude_range,
                durs=delta_duration_range,
                msr=result.magnitude,
                phase=result.phase,
                iq_distance=iq_distance,
            )

    return data


def _fit(data: ChevronFluxTimeData):
    return ChevronFluxTimeResults()


def _plot(data: ChevronFluxTimeData, fit: ChevronFluxTimeResults, qubit):
    for q in qubit:
        if q != 2:
            coupler = q
    pair = qubit

    states = ["low", "high"]
    # FIXME: Get qubits
    # pair = sort_frequency(pair)
    titles = [
        "low_MSR",
        "high_MSR",
        "low_phase",
        "high_phase",
        "low_iq distance",
        "high_iq distance",
    ]
    figures = []
    fig = make_subplots(
        rows=3, cols=2, shared_xaxes=True, shared_yaxes=True, subplot_titles=titles
    )
    colouraxis = ["coloraxis", "coloraxis2"]
    fitting_report = "No fitting data"
    labels = ["MSR", "phase", "iq_distance"]

    for state, q in zip(states, pair):
        # FIXME: Get qubits
        # for x in pair.qubit1.flux_coupler.keys():
        #     if x in pair.qubit2.flux_coupler.keys():
        #         coupler = x
        #         break

        duration = data[q, coupler, state].duration
        amplitude = data[q, coupler, state].amplitude
        for values, label in zip(
            [
                data[q, coupler, state].msr,
                data[q, coupler, state].phase,
                data[q, coupler, state].iq_distance,
            ],
            labels,
        ):
            fig.add_trace(
                go.Heatmap(
                    y=duration,
                    x=amplitude,
                    z=values,
                    name=f"Qubit {q} |{state}>",
                    coloraxis=colouraxis[states.index(state)],
                ),
                row=labels.index(label) + 1,
                col=states.index(state) + 1,
            )

        fig.update_layout(
            coloraxis=dict(colorscale="Viridis", colorbar=dict(x=0.45)),
            coloraxis2=dict(colorscale="Cividis", colorbar=dict(x=1)),
        )

    fig.update_layout(
        title=f"Qubit {coupler} Interaction {label}",
        yaxis_title="Duration [ns]",
        yaxis3_title="Duration [ns]",
        yaxis5_title="Duration [ns]",
        xaxis5_title="Amplitude [dimensionless]",
        xaxis6_title="Amplitude [dimensionless]",
        legend_title="States",
    )

    figures.append(fig)
    return figures, fitting_report


def sort_frequency(pair: QubitPair):
    """Sorts the qubits in a pair by frequency."""
    if pair.qubit1.drive_frequency > pair.qubit2.drive_frequency:
        return QubitPair(pair.qubit2, pair.qubit1)
    else:
        return QubitPair(pair.qubit1, pair.qubit2)


chevron_flux_time = Routine(_aquisition, _fit, _plot)
"""Coupler cz/swap flux routine."""
