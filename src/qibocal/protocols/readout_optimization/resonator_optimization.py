import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import scipy.ndimage as ndimage
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    Delay,
    Parameter,
    PulseSequence,
    Readout,
    Sweeper,
)

from qibocal import update
from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.fitting.classifier.qubit_fit import QubitFit
from qibocal.protocols.utils import HZ_TO_GHZ, readout_frequency, table_dict, table_html


@dataclass
class ResonatorOptimizationParameters(Parameters):
    """Resonator optimization runcard inputs"""

    freq_width: int
    """Width for frequency sweep relative  to the readout frequency [Hz]."""
    freq_step: int
    """Frequency step for sweep [Hz]."""
    amplitude_start: float
    """Minimum amplitude."""
    amplitude_stop: float
    """Maximum amplitude."""
    amplitude_step: float
    """Step amplitude."""
    delay: float = 0
    """Delay between readouts, could account for resonator deplation or not [ns]."""


@dataclass
class ResonatorOptimizationResults(Results):
    """Resonator optimization outputs"""

    best_fidelity: dict[QubitId, list]
    """Best assignment fidelities."""
    fid_best_freq: dict[QubitId, float]
    """Resonator Frequency with the highest assignment fidelity."""
    fid_best_amp: dict[QubitId, list]
    """Resonator Amplitude with the highest assignment fidelity"""
    fid_best_qnd: dict[QubitId, list]
    """Quantum Non Demolition-ness with the highest assignment fidelity."""
    best_qnd: dict[QubitId, list]
    """Best quantum non demolition-ness."""
    qnd_best_freq: dict[QubitId, list]
    """Resonator Frequency with the highest quantum non demolition-ness."""
    qnd_best_amp: dict[QubitId, list]
    """Resonator Amplittude with the highest quantum non demolition-ness."""
    qnd_best_fid: dict[QubitId, list]
    """Fidelity with highest quantum non demolition-ness"""
    best_angle: dict[QubitId, float]
    """IQ angle that maximes assignment fidelity."""
    best_threshold: dict[QubitId, float]
    """Threshold that maximes assignment fidelity."""


ResonatorOptimizationType = np.dtype(
    [
        ("frequency", np.float64),
        ("amplitude", np.float64),
        ("iq_values", np.float64, (2,)),
        ("assignment_fidelity", np.float64),
        ("avaraged_fidelity", np.float64),
        ("qnd", np.float64),
        ("angle", np.float64),
        ("threshold", np.float64),
    ]
)
"""Custom dtype readout optimization."""


@dataclass
class ResonatorOptimizationData(Data):
    """Data class for resonator optimization protocol."""

    resonator_type: str
    """Resonator type."""
    delay: float = 0
    """Delay between readouts [ns]."""
    amplitudes: dict[QubitId, float] = field(default_factory=dict)
    """Amplitudes provided by the user."""
    qubit_frequencies: dict[QubitId, float] = field(default_factory=dict)
    """Qubit frequencies."""
    data: dict[QubitId, npt.NDArray[ResonatorOptimizationType]] = field(
        default_factory=dict
    )
    """Raw data acquired"""
    samples: dict[tuple, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: ResonatorOptimizationParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> ResonatorOptimizationData:
    r"""
    Data acquisition for readout optimization.

    Args:
        params (ResonatorFrequencyParameters): experiment's parameters
        platform (Platform): Qibolab platform object
        qubits (list): list of target qubits to perform the action
    """
    delta_frequency_range = np.arange(
        -params.freq_width / 2, params.freq_width / 2, params.freq_step
    )

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    amplitudes = {}
    freq_sweepers = {}
    ro_pulses_m1 = {}
    ro_pulses_m2 = {}

    data = ResonatorOptimizationData(
        resonator_type=platform.resonator_type,
        delay=params.delay,
        qubit_frequencies={
            # TODO: should this be the drive frequency instead?
            qubit: float(platform.calibration.single_qubits[qubit].qubit.frequency_01)
            for qubit in targets
        },
    )

    for state in [0, 1]:
        sequence = PulseSequence()
        for qubit in targets:
            natives = platform.natives.single_qubit[qubit]
            ro_channel, ro_pulse_m1 = natives.MZ()[0]
            _, ro_pulse_m2 = natives.MZ()[0]
            if state == 1:
                sequence += natives.RX()
            sequence.append((ro_channel, Delay(duration=natives.RX()[0][1].duration)))
            sequence.append((ro_channel, ro_pulse_m1))
            sequence.append((ro_channel, Delay(duration=params.delay)))
            sequence.append((ro_channel, ro_pulse_m2))

            amplitudes[qubit] = ro_pulse_m1.probe.amplitude
            data.amplitudes = amplitudes

            freq_sweepers[qubit] = Sweeper(
                parameter=Parameter.frequency,
                values=readout_frequency(qubit, platform) + delta_frequency_range,
                channels=[platform.qubits[qubit].probe],
            )

            ro_pulses_m1[qubit] = ro_pulse_m1
            ro_pulses_m2[qubit] = ro_pulse_m2

        amp_sweeper = Sweeper(
            parameter=Parameter.amplitude,
            range=(
                params.amplitude_start,
                params.amplitude_stop,
                params.amplitude_step,
            ),
            pulses=[ro_pulses_m1[qubit] for qubit in targets]
            + [ro_pulses_m2[qubit] for qubit in targets],
        )

        results = platform.execute(
            [sequence],
            [[amp_sweeper], [freq_sweepers[q] for q in targets]],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
        )
        results_samples = platform.execute(
            [sequence],
            [[amp_sweeper], [freq_sweepers[q] for q in targets]],
            acquisition_type=AcquisitionType.DISCRIMINATION,
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
        )

        for target in targets:
            readouts = [
                pulse
                for pulse in sequence.channel(platform.qubits[qubit].acquisition)
                if isinstance(pulse, Readout)
            ]

            for n in np.arange(0, params.nshots, 1):
                for j, amp in enumerate(amp_sweeper.values):
                    for k, freq in enumerate(freq_sweepers[qubit].values):
                        for m, ro_pulse in enumerate(readouts):
                            # m signal if it's the first or second measurment for QND
                            data.register_qubit(
                                ResonatorOptimizationType,
                                (qubit, state, m),
                                dict(
                                    frequency=np.array([freq]),
                                    amplitude=np.array([amp]),
                                    iq_values=np.array([results[ro_pulse.id][n][j][k]]),
                                ),
                            )
                            data.samples[qubit, state, m] = results_samples[
                                ro_pulse.id
                            ].tolist()

    return data


def _fit(data: ResonatorOptimizationData) -> ResonatorOptimizationResults:
    qubits = data.qubits

    fid_best_freq = {}
    fid_best_amps = {}
    fid_best_qnd = {}
    best_angle = {}
    best_threshold = {}
    highest_fidelity = {}
    highest_qnd = {}
    qnd_best_freq = {}
    qnd_best_amps = {}
    qnd_best_fid = {}
    Lambda_M = {}
    Lambda_M2 = {}

    for qubit in qubits:
        for state, m in itertools.product([0, 1], [0, 1]):
            freq_vals = np.unique(data[qubit, state, m]["frequency"])
            amp_vals = np.unique(data[qubit, state, m]["amplitude"])

        fidelity_grid = np.zeros(shape=(len(freq_vals), len(amp_vals)))
        qnd_grid = np.zeros(shape=(len(freq_vals), len(amp_vals)))
        angle_grid = np.zeros(shape=(len(freq_vals), len(amp_vals)))
        threshold_grid = np.zeros(shape=(len(freq_vals), len(amp_vals)))

        ################################ASSIGNMENT FIDELITY################################

        for j, freq in enumerate(freq_vals):
            for k, amp in enumerate(amp_vals):
                data_state_0 = data[qubit, 0, 0]
                data_state_1 = data[qubit, 1, 0]
                iq_values = np.concatenate(
                    (
                        data_state_0[
                            (data_state_0.frequency == freq)
                            & (data_state_0.amplitude == amp)
                        ].iq_values,
                        data_state_1[
                            (data_state_1.frequency == freq)
                            & (data_state_1.amplitude == amp)
                        ].iq_values,
                    )
                )

                nshots = len(
                    data_state_0[
                        (data_state_0.frequency == freq)
                        & (data_state_0.amplitude == amp)
                    ].iq_values
                )
                states = [0] * nshots + [1] * nshots

                model = QubitFit()
                model.fit(iq_values, np.array(states))
                fidelity_grid[j, k] = model.assignment_fidelity
                angle_grid[j, k] = model.angle
                threshold_grid[j, k] = model.threshold

        filtered_fidelity = ndimage.uniform_filter(
            fidelity_grid,
            size=(math.ceil(len(freq_vals) / 5), math.ceil(len(amp_vals) / 5)),
            mode="nearest",
        )

        ######################################## QND ######################################
        for j, freq in enumerate(freq_vals):
            for k, amp in enumerate(amp_vals):
                # 1st measurement (m=1)
                m1_state_1 = data.samples[qubit, 1, 0]
                nshots = len(m1_state_1)
                # state 1
                state1_count_1_m1 = np.count_nonzero(m1_state_1)
                state0_count_1_m1 = nshots - state1_count_1_m1

                m1_state_0 = data.samples[qubit, 0, 0]
                # state 0
                state1_count_0_m1 = np.count_nonzero(m1_state_0)
                state0_count_0_m1 = nshots - state1_count_0_m1

                # 2nd measurement (m=2)
                m2_state_1 = data.samples[qubit, 1, 1]
                # state 1
                state1_count_1_m2 = np.count_nonzero(m2_state_1)
                state0_count_1_m2 = nshots - state1_count_1_m2

                m2_state_0 = data.samples[qubit, 0, 1]
                # state 0
                state1_count_0_m2 = np.count_nonzero(m2_state_0)
                state0_count_0_m2 = nshots - state1_count_0_m2

                # Repeat Lambda and fidelity for each measurement ?
                Lambda_M[qubit] = [
                    [state0_count_0_m1 / nshots, state0_count_1_m1 / nshots],
                    [state1_count_0_m1 / nshots, state1_count_1_m1 / nshots],
                ]

                # Repeat Lambda and fidelity for each measurement ?
                Lambda_M2[qubit] = [
                    [state0_count_0_m2 / nshots, state0_count_1_m2 / nshots],
                    [state1_count_0_m2 / nshots, state1_count_1_m2 / nshots],
                ]

                # QND FIXME: Careful revision
                P_0o_m0_1i = state0_count_1_m1 * state0_count_0_m2 / nshots**2
                P_0o_m1_1i = state1_count_1_m1 * state0_count_1_m2 / nshots**2
                P_0o_1i = P_0o_m0_1i + P_0o_m1_1i

                P_1o_m0_0i = state0_count_0_m1 * state1_count_0_m2 / nshots**2
                P_1o_m1_0i = state1_count_0_m1 * state1_count_1_m2 / nshots**2
                P_1o_0i = P_1o_m0_0i + P_1o_m1_0i

                qnd_grid[j, k] = 1 - (P_0o_1i + P_1o_0i) / 2

        for state, m in itertools.product([0, 1], [0, 1]):
            data_qubit = data[qubit, state, m]

            for j, freq in enumerate(freq_vals):
                for k, amp in enumerate(amp_vals):
                    update_index = (data_qubit.frequency == freq) & (
                        data_qubit.amplitude == amp
                    )
                    data_qubit.avaraged_fidelity[update_index] = filtered_fidelity[j, k]
                    data_qubit.angle[update_index] = angle_grid[j, k]
                    data_qubit.threshold[update_index] = threshold_grid[j, k]
                    data_qubit.qnd[update_index] = qnd_grid[j, k]

        index_best_fid = np.argmax(data[qubit, 0, 0]["avaraged_fidelity"])
        highest_fidelity[qubit] = data[qubit, 0, 0]["avaraged_fidelity"][index_best_fid]
        fid_best_freq[qubit] = data[qubit, 0, 0]["frequency"][index_best_fid]
        fid_best_amps[qubit] = data[qubit, 0, 0]["amplitude"][index_best_fid]
        fid_best_qnd[qubit] = data[qubit, 0, 0]["qnd"][index_best_fid]
        best_angle[qubit] = data[qubit, 0, 0]["angle"][index_best_fid]
        best_threshold[qubit] = data[qubit, 0, 0]["threshold"][index_best_fid]

        index_best_qnd = np.argmax(data[qubit, 0, 0]["qnd"])
        highest_qnd[qubit] = data[qubit, 0, 0]["qnd"][index_best_qnd]
        qnd_best_freq[qubit] = data[qubit, 0, 0]["frequency"][index_best_qnd]
        qnd_best_amps[qubit] = data[qubit, 0, 0]["amplitude"][index_best_qnd]
        qnd_best_fid[qubit] = data[qubit, 0, 0]["avaraged_fidelity"][index_best_qnd]

    return ResonatorOptimizationResults(
        best_fidelity=highest_fidelity,
        fid_best_freq=fid_best_freq,
        fid_best_amp=fid_best_amps,
        fid_best_qnd=fid_best_qnd,
        best_qnd=highest_qnd,
        qnd_best_freq=qnd_best_freq,
        qnd_best_amp=qnd_best_amps,
        qnd_best_fid=qnd_best_fid,
        best_angle=best_angle,
        best_threshold=best_threshold,
    )


def _plot(
    data: ResonatorOptimizationData, fit: ResonatorOptimizationResults, target: QubitId
):
    """Plotting function for resonator optimization"""

    qubit_data = data[target, 0, 0]
    figures = []
    fitting_report = ""

    fig = make_subplots(
        rows=1,
        cols=2,
    )

    frequencies = qubit_data.frequency
    amplitudes = qubit_data.amplitude
    fidelities = qubit_data.avaraged_fidelity
    qnds = qubit_data.qnd

    if fit is not None:
        fig.add_trace(
            go.Heatmap(
                x=amplitudes,
                y=frequencies * HZ_TO_GHZ,
                z=fidelities,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[fit.fid_best_amp],
                y=[fit.fid_best_freq],
                mode="markers",
                marker=dict(
                    size=8,
                    color="black",
                    symbol="cross",
                ),
                name="highest assignment fidelity",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Heatmap(
                x=amplitudes,
                y=frequencies * HZ_TO_GHZ,
                z=qnds,
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=[fit.qnd_best_amp],
                y=[fit.qnd_best_freq],
                mode="markers",
                marker=dict(
                    size=8,
                    color="black",
                    symbol="cross",
                ),
                name="highest quantum non demolition-ness",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        fitting_report = table_html(
            table_dict(
                target,
                [
                    "Best Assignment-Fidlity Amplitude [a.u.]",
                    "Best Assignment-Fidlity Frequency [GHz]",
                    "Best Assignment-Fidlity",
                    "Best QND Amplitude [a.u.]",
                    "Best QND Frequency [GHz]",
                    "Best Quantum Non Demolition-ness",
                ],
                [
                    np.round(fit.fid_best_amp[target], 4),
                    np.round(fit.fid_best_freq[target]) * HZ_TO_GHZ,
                    fit.best_fidelity[target],
                    np.round(fit.qnd_best_amp[target], 4),
                    np.round(fit.qnd_best_freq[target]) * HZ_TO_GHZ,
                    fit.best_qnd[target],
                ],
            )
        )

        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h"),
        )

        fig.update_xaxes(title_text="Amplitude [a.u.]", row=1, col=1)
        fig.update_yaxes(title_text="Frequency [GHz]", row=1, col=1)

        figures.append(fig)
    return figures, fitting_report


def _update(
    results: ResonatorOptimizationResults,
    platform: CalibrationPlatform,
    target: QubitId,
):
    update.readout_amplitude(results.fid_best_amp[target], platform, target)
    update.readout_frequency(results.fid_best_freq[target], platform, target)
    update.iq_angle(results.best_angle[target], platform, target)
    update.threshold(results.best_threshold[target], platform, target)


resonator_optimization = Routine(
    _acquisition,
    _fit,
    _plot,
    _update,
)
"""Resonator optimization Routine object"""
