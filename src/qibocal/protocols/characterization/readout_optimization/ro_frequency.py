from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine


@dataclass
class RoFrequencyParameters(Parameters):
    """Dispersive shift inputs."""

    freq_width: int
    """Width [Hz] for frequency sweep relative to the readout frequency (Hz)."""
    freq_step: int
    """Frequency step for sweep (Hz)."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class RoFrequencyResults(Results):
    """Dispersive shift outputs."""

    fidelities: dict[QubitId, list]

    best_freq: dict[QubitId, float] = field(metadata=dict(update="readout_frequency"))


RoFrequencyType = np.dtype(
    [
        ("freq", np.float64),
        ("i", np.float64),
        ("q", np.float64),
        ("msr", np.float64),
        ("phase", np.float64),
    ]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class RoFrequencyData(Data):
    """Dipsersive shift acquisition outputs."""

    resonator_type: str
    """Resonator type."""
    data: dict[tuple[QubitId, int, int], npt.NDArray[RoFrequencyType]] = field(
        default_factory=dict
    )

    def register_qubit(self, qubit, state, freq, msr, phase, i, q):
        """Store output for single qubit."""
        ar = np.empty(i.shape, dtype=RoFrequencyType)
        ar["freq"] = freq
        ar["msr"] = msr
        ar["phase"] = phase
        ar["i"] = i
        ar["q"] = q
        self.data[qubit, state] = np.rec.array(ar)


def _acquisition(
    params: RoFrequencyParameters, platform: Platform, qubits: Qubits
) -> RoFrequencyData:
    r"""
    Data acquisition for dispersive shift experiment.
    Perform spectroscopy on the readout resonator, with the qubit in ground and excited state, showing
    the resonator shift produced by the coupling between the resonator and the qubit.

    Args:
        params (RoFrequencyParameters): experiment's parameters
        platform (Platform): Qibolab platform object
        qubits (dict): list of target qubits to perform the action

    """

    # create 2 sequences of pulses for the experiment:
    # sequence_0: I  - MZ
    # sequence_1: RX - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence_0 = PulseSequence()
    sequence_1 = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence_0.add(ro_pulses[qubit])
        sequence_1.add(qd_pulses[qubit])
        sequence_1.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(
        -params.freq_width / 2, params.freq_width / 2, params.freq_step
    )

    data = RoFrequencyData(platform.resonator_type)
    sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[ro_pulses[qubit] for qubit in qubits],
        type=SweeperType.OFFSET,
    )

    results_0 = platform.sweep(
        sequence_0,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            # averaging_mode=AveragingMode.SINGLESHOT,
        ),
        sweeper,
    )

    results_1 = platform.sweep(
        sequence_1,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            # averaging_mode=AveragingMode.SINGLESHOT,
        ),
        sweeper,
    )

    # retrieve the results for every qubit
    for qubit in qubits:
        # average msr, phase, i and q over the number of shots defined in the runcard
        for i, results in enumerate([results_0, results_1]):
            result = results[ro_pulses[qubit].serial]
            # store the results
            data.register_qubit(
                qubit=qubit,
                state=i,
                freq=ro_pulses[qubit].frequency + delta_frequency_range,
                msr=result.magnitude,
                phase=result.phase,
                i=result.voltage_i,
                q=result.voltage_q,
            )
    return data


def _fit(data: RoFrequencyData) -> RoFrequencyResults:
    """Post-Processing for dispersive shift"""
    qubits = data.qubits
    freqs = np.unique(data[qubits[0], 0].freq)
    fidelities_dict = {}
    fidelities = []
    best_freqs = {}
    print(freqs)
    for qubit in qubits:
        for freq in freqs:
            iq_state0 = data[qubit, 0][data[qubit, 0].freq == freq][["i", "q"]]
            iq_state1 = data[qubit, 1][data[qubit, 1].freq == freq][["i", "q"]]

            import matplotlib.pyplot as plt

            plt.subplots()
            plt.scatter(iq_state0.i, iq_state0.q)
            plt.scatter(iq_state1.i, iq_state1.q)

            iq_state0 = iq_state0.i + 1.0j * iq_state0.q
            iq_state1 = iq_state1.i + 1.0j * iq_state1.q

            iq_state1 = np.array(iq_state1)
            iq_state0 = np.array(iq_state0)
            nshots = len(iq_state0)
            # print(iq_state0, iq_state1)

            iq_mean_state1 = np.mean(iq_state1)
            iq_mean_state0 = np.mean(iq_state0)

            vector01 = iq_mean_state1 - iq_mean_state0
            rotation_angle = np.angle(vector01)

            iq_state1_rotated = iq_state1 * np.exp(-1j * rotation_angle)
            iq_state0_rotated = iq_state0 * np.exp(-1j * rotation_angle)

            real_values_state1 = iq_state1_rotated.real
            real_values_state0 = iq_state0_rotated.real

            real_values_combined = np.concatenate(
                (real_values_state1, real_values_state0)
            )
            real_values_combined.sort()

            cum_distribution_state1 = [
                sum(map(lambda x: x.real >= real_value, real_values_state1))
                for real_value in real_values_combined
            ]
            cum_distribution_state0 = [
                sum(map(lambda x: x.real >= real_value, real_values_state0))
                for real_value in real_values_combined
            ]

            cum_distribution_diff = np.abs(
                np.array(cum_distribution_state1) - np.array(cum_distribution_state0)
            )
            argmax = np.argmax(cum_distribution_diff)
            errors_state1 = nshots - cum_distribution_state1[argmax]
            errors_state0 = cum_distribution_state0[argmax]
            assignment_fidelity = 1 - (errors_state1 + errors_state0) / nshots / 2
            print("AAAAAAAAAAAAA ", nshots, assignment_fidelity)
            fidelities.append(assignment_fidelity)
            plt.title(f"{assignment_fidelity}")
            plt.savefig(f"PLOOOOT{freq}.pdf")
        fidelities_dict[qubit] = fidelities

        best_freqs[qubit] = freqs[np.argmax(fidelities_dict[qubit])]
        print(fidelities_dict, np.argmax(fidelities_dict[qubit]), best_freqs)
    # results = []
    # iq_couples = [[], []]  # axis 0: states, axis 1: qubit
    # for i in range(2):
    #     frequency = {}
    #     fitted_parameters = {}
    #     for qubit in qubits:
    #         data_i = data[qubit, i]
    #         freq, fitted_params = lorentzian_fit(
    #             data_i, resonator_type=data.resonator_type, fit="resonator"
    #         )
    #         frequency[qubit] = freq
    #         fitted_parameters[qubit] = fitted_params
    #         i_measures = data_i.i
    #         q_measures = data_i.q

    #         iq_couples[i].append(np.stack((i_measures, q_measures), axis=-1))
    #         results.append(StateResults(frequency, fitted_parameters))

    # # for each qubit find the iq couple of 0-1 states that maximize the distance
    # iq_couples = np.array(iq_couples)
    # best_freqs = {}
    # best_iqs = {}
    # for qubit in qubits:
    #     frequencies = data[qubit, 0].freq * HZ_TO_GHZ

    #     max_index = np.argmax(
    #         np.linalg.norm(iq_couples[0][qubit] - iq_couples[1][qubit], axis=-1)
    #     )
    #     best_freqs[qubit] = frequencies[max_index]
    #     best_iqs[qubit] = iq_couples[:, qubit, max_index].tolist()

    return RoFrequencyResults(
        fidelities=fidelities_dict,
        best_freq=best_freqs,
    )


def _plot(data: RoFrequencyData, fit: RoFrequencyResults, qubit):
    """Plotting function for dispersive shift."""
    figures = []
    freqs = np.unique(data[qubit, 0].freq)
    opacity = 1
    fitting_report = " "
    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=("RES FREQ",),
    )

    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=fit.fidelities[qubit],
            opacity=opacity,
            # name=f"q{qubit}: {label}",
            showlegend=True,
            # legendgroup=f"q{qubit}: {label}",
        ),
        row=1,
        col=1,
    )

    # fig = make_subplots(
    #     rows=1,
    #     cols=2,
    #     horizontal_spacing=0.1,
    #     vertical_spacing=0.1,
    #     subplot_titles=(
    #         "MSR (V)",
    #         "phase (rad)",
    #     ),
    # )
    # # iterate over multiple data folders

    # fitting_report = ""

    # data_0 = data[qubit, 0]
    # data_1 = data[qubit, 1]

    # fit_data_0 = fit.results_0
    # fit_data_1 = fit.results_1

    # for i, label, q_data, data_fit in list(
    #     zip(
    #         (0, 1),
    #         ("State 0", "State 1"),
    #         (data_0, data_1),
    #         (fit_data_0, fit_data_1),
    #     )
    # ):
    #     opacity = 1
    #     frequencies = q_data.freq * HZ_TO_GHZ

    #     fig.add_trace(
    #         go.Scatter(
    #             x=frequencies,
    #             y=q_data.msr * V_TO_UV,
    #             opacity=opacity,
    #             name=f"q{qubit}: {label}",
    #             showlegend=True,
    #             legendgroup=f"q{qubit}: {label}",
    #         ),
    #         row=1,
    #         col=1,
    #     )
    #     fig.add_trace(
    #         go.Scatter(
    #             x=frequencies,
    #             y=q_data.phase,
    #             opacity=opacity,
    #             showlegend=False,
    #             legendgroup=f"q{qubit}: {label}",
    #         ),
    #         row=1,
    #         col=2,
    #     )

    #     freqrange = np.linspace(
    #         min(frequencies),
    #         max(frequencies),
    #         2 * len(q_data),
    #     )

    #     params = data_fit.fitted_parameters[qubit]
    #     fig.add_trace(
    #         go.Scatter(
    #             x=freqrange,
    #             y=lorentzian(freqrange, **params),
    #             name=f"q{qubit}: {label} Fit",
    #             line=go.scatter.Line(dash="dot"),
    #         ),
    #         row=1,
    #         col=1,
    #     )

    # fig.add_trace(
    #     go.Scatter(
    #         x=[fit.best_freq[qubit], fit.best_freq[qubit]],
    #         y=[
    #             np.min(np.concatenate((data_0.msr, data_1.msr))),
    #             np.max(np.concatenate((data_0.msr, data_1.msr))),
    #         ],
    #         mode="lines",
    #         line=go.scatter.Line(color="orange", width=3, dash="dash"),
    #         name="Best frequency",
    #     ),
    #     row=1,
    #     col=1,
    # )

    # fig.add_vline(
    #     x=fit.best_freq[qubit],
    #     line=dict(color="orange", width=3, dash="dash"),
    #     row=1,
    #     col=1,
    # )

    fitting_report = fitting_report + (
        f"{qubit} | State zero freq : {0.03:,.0f} Hz.<br>"
    )
    # fitting_report = fitting_report + (
    #     f"{qubit} | State one freq : {fit_data_1.frequency[qubit]*GHZ_TO_HZ:,.0f} Hz.<br>"
    # )
    # fitting_report = fitting_report + (
    #     f"{qubit} | Chi : {(fit_data_0.frequency[qubit]*GHZ_TO_HZ - fit_data_1.frequency[qubit]*GHZ_TO_HZ)/2:,.0f} Hz.<br>"
    # )
    # fitting_report = fitting_report + (
    #     f"{qubit} | Best frequency : {fit.best_freq[qubit]*GHZ_TO_HZ:,.0f} Hz.<br>"
    # )
    # fig.update_layout(
    #     showlegend=True,
    #     uirevision="0",  # ``uirevision`` allows zooming while live plotting
    #     xaxis_title="Frequency (GHz)",
    #     yaxis_title="MSR (uV)",
    #     xaxis2_title="Frequency (GHz)",
    #     yaxis2_title="Phase (rad)",
    # )

    figures.append(fig)

    return figures, fitting_report


ro_frequency = Routine(_acquisition, _fit, _plot)
"""Dispersive shift Routine object."""
