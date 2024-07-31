from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Routine
from qibocal.protocols.utils import HZ_TO_GHZ, lorentzian

from .dispersive_shift import (
    DispersiveShiftData,
    DispersiveShiftParameters,
    DispersiveShiftResults,
    DispersiveShiftType,
)


@dataclass
class DispersiveShiftRestlessParameters(DispersiveShiftParameters):
    """Dispersive shift inputs."""

    delay_start: float
    """Initial delay parameter for resonator depletion."""
    delay_end: float
    """Final delay parameter for resonator depletion."""
    delay_step: float
    """Step delay parameter for resonator depletion."""


@dataclass
class DispersiveShiftRestlessData(DispersiveShiftData):
    """Dipsersive shift acquisition outputs."""

    @property
    def delays(self):
        """Delay parameter for resonator depletion."""
        return np.unique([b[1] for b in self.data])


def _acquisition(
    params: DispersiveShiftRestlessParameters,
    platform: Platform,
    targets: list[QubitId],
) -> DispersiveShiftRestlessData:
    r"""
    Data acquisition for dispersive shift experiment.
    Perform spectroscopy on the readout resonator, with the qubit in ground and excited state, showing
    the resonator shift produced by the coupling between the resonator and the qubit.

    Args:
        params (DispersiveShiftParameters): experiment's parameters
        platform (Platform): Qibolab platform object
        targets (list): list of target qubits to perform the action

    """

    # create 2 sequences of pulses for the experiment:
    # sequence_0: I  - MZ
    # sequence_1: RX - MZ

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(
        -params.freq_width / 2, params.freq_width / 2, params.freq_step
    )
    delays = np.arange(params.delay_start, params.delay_end, params.delay_step)
    data = DispersiveShiftRestlessData(resonator_type=platform.resonator_type)

    for delay in delays:
        sequence_0 = PulseSequence()
        sequence_1 = PulseSequence()

        for qubit in targets:

            ro_pulse_0 = platform.create_qubit_readout_pulse(qubit, start=0)
            sequence_0.add(ro_pulse_0)
            sequence_1.add(ro_pulse_0)

            rx_pulse = platform.create_RX_pulse(qubit, start=delay)
            sequence_1.add(rx_pulse)

            ro_pulse = platform.create_qubit_readout_pulse(qubit, start=delay)
            sequence_0.add(ro_pulse)

            ro_pulse = platform.create_qubit_readout_pulse(
                qubit, start=sequence_1.qd_pulses.finish
            )
            sequence_1.add(ro_pulse)

        for state, sequence in enumerate([sequence_0, sequence_1]):
            sweeper = Sweeper(
                Parameter.frequency,
                delta_frequency_range,
                pulses=list(sequence.ro_pulses),
                type=SweeperType.OFFSET,
            )

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

            for qubit in targets:
                result = results[qubit].average
                # store the results
                data.register_qubit(
                    DispersiveShiftType,
                    (qubit, float(delay), state),
                    dict(
                        freq=sequence.get_qubit_pulses(qubit).ro_pulses[1].frequency
                        + delta_frequency_range,
                        signal=result.magnitude,
                        phase=result.phase,
                        i=result.voltage_i,
                        q=result.voltage_q,
                    ),
                )

    return data


def _fit(data: DispersiveShiftData) -> DispersiveShiftResults:
    """Post-Processing for dispersive shift"""
    qubits = data.qubits
    iq_couples = [[], []]  # axis 0: states, axis 1: qubit

    frequency_0 = {}
    frequency_1 = {}
    best_freqs = {}
    fitted_parameters_0 = {}
    fitted_parameters_1 = {}

    # for delay in data.delays:
    #     for i in range(2):
    #         for qubit in qubits:
    #             data_i = data[qubit, delay, i]
    #             fit_result = lorentzian_fit(
    #                 data_i, resonator_type=data.resonator_type, fit="resonator"
    #             )
    #             if fit_result is not None:
    #                 if i == 0:
    #                     frequency_0[qubit, delay], fitted_parameters_0[qubit, delay], _ = fit_result
    #                 else:
    #                     frequency_1[qubit, delay], fitted_parameters_1[qubit, delay], _ = fit_result

    #             i_measures = data_i.i
    #             q_measures = data_i.q

    #             iq_couples[i].append(np.stack((i_measures, q_measures), axis=-1))
    #         # for each qubit find the iq couple of 0-1 states that maximize the distance
    #     iq_couples = np.array(iq_couples)

    #     for idx, qubit in enumerate(qubits):
    #         frequencies = data[qubit, delay, 0].freq

    #         max_index = np.argmax(
    #             np.linalg.norm(iq_couples[0][idx] - iq_couples[1][idx], axis=-1)
    #         )
    #         best_freqs[qubit, delay] = frequencies[max_index]

    return DispersiveShiftResults(
        frequency_state_zero=frequency_0,
        frequency_state_one=frequency_1,
        fitted_parameters_state_one=fitted_parameters_1,
        fitted_parameters_state_zero=fitted_parameters_0,
        best_freq=best_freqs,
    )


def _plot(
    data: DispersiveShiftRestlessData, target: QubitId, fit: DispersiveShiftResults
):
    """Plotting function for dispersive shift."""
    figures = []
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "Signal [a.u.]",
            "phase [rad]",
        ),
    )
    # iterate over multiple data folders

    fitting_report = ""

    for j, delay in enumerate(data.delays):
        data_0 = data[target, delay, 0]
        data_1 = data[target, delay, 1]
        fit_data_0 = fit.state_zero if fit is not None else None
        fit_data_1 = fit.state_one if fit is not None else None

        for i, label, q_data, data_fit in list(
            zip(
                (0, 1),
                ("State 0 Delay" + str(delay), "State 1 Delay" + str(delay)),
                (data_0, data_1),
                (fit_data_0, fit_data_1),
            )
        ):
            opacity = 1
            frequencies = q_data.freq * HZ_TO_GHZ
            fig.add_trace(
                go.Scatter(
                    x=frequencies,
                    y=q_data.signal,
                    opacity=opacity,
                    name=f"{label}",
                    showlegend=True,
                    legendgroup=f"{label}",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=frequencies,
                    y=q_data.phase,
                    opacity=opacity,
                    showlegend=False,
                    legendgroup=f"{label}",
                ),
                row=1,
                col=2,
            )

            if fit is not None:
                freqrange = np.linspace(
                    min(frequencies),
                    max(frequencies),
                    2 * len(q_data),
                )
                params = data_fit[
                    (
                        "fitted_parameters_state_zero"
                        if i == 0
                        else "fitted_parameters_state_one"
                    )
                ][target]
                fig.add_trace(
                    go.Scatter(
                        x=freqrange,
                        y=lorentzian(freqrange, *params),
                        name=f"{label} Fit",
                        line=go.scatter.Line(dash="dot"),
                        legendgroup=f"group{j}",
                        showlegend=True,
                    ),
                    row=1,
                    col=1,
                )

        if fit is not None:
            fig.add_trace(
                go.Scatter(
                    x=[
                        fit.best_freq[target, delay] * HZ_TO_GHZ,
                        fit.best_freq[target, delay] * HZ_TO_GHZ,
                    ],
                    y=[
                        np.min(np.concatenate((data_0.signal, data_1.signal))),
                        np.max(np.concatenate((data_0.signal, data_1.signal))),
                    ],
                    mode="lines",
                    line=go.scatter.Line(color="orange", width=3, dash="dash"),
                    name="Best frequency",
                ),
                row=1,
                col=1,
            )

            fig.add_vline(
                x=fit.best_freq[target, delay] * HZ_TO_GHZ,
                line=dict(color="orange", width=3, dash="dash"),
                row=1,
                col=1,
            )
        # fitting_report = table_html(
        #     table_dict(
        #         target,
        #         [
        #             "State Zero Frequency [Hz]",
        #             "State One Frequency [Hz]",
        #             "Chi [Hz]",
        #             "Best Frequency [Hz]",
        #         ],
        #         np.round(
        #             [
        #                 fit_data_0["frequency_state_zero"][target],
        #                 fit_data_1["frequency_state_one"][target],
        #                 (
        #                     fit_data_0["frequency_state_zero"][target]
        #                     - fit_data_1["frequency_state_one"][target]
        #                 )
        #                 / 2,
        #                 fit.best_freq[target],
        #             ]
        #         ),
        #     )
        # )
    fig.update_layout(
        showlegend=True,
        xaxis_title="Frequency [GHz]",
        yaxis_title="Signal [a.u.]",
        xaxis2_title="Frequency [GHz]",
        yaxis2_title="Phase [rad]",
    )

    figures.append(fig)

    return figures, fitting_report


dispersive_shift_restless = Routine(_acquisition, _fit, _plot)
