from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
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

    fidelities: dict[QubitId, list]
    """Assignment fidelities."""
    best_freq: dict[QubitId, float]
    """Resonator Frequency with the highest assignment fidelity."""
    best_amp: dict[QubitId, list]
    """Amplitude with lowest error."""
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
    ro_pulses = {}

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
            ro_channel, ro_pulse = natives.MZ()[0]
            if state == 1:
                sequence += natives.RX()
            sequence.append((ro_channel, Delay(duration=natives.RX()[0][1].duration)))
            sequence += natives.MZ()
            sequence.append((ro_channel, Delay(duration=params.delay)))
            sequence += natives.MZ()

            amplitudes[qubit] = ro_pulse.probe.amplitude
            data.amplitudes = amplitudes

            freq_sweepers[qubit] = Sweeper(
                parameter=Parameter.frequency,
                values=readout_frequency(qubit, platform) + delta_frequency_range,
                channels=[platform.qubits[qubit].probe],
            )

            ro_pulses[qubit] = ro_pulse

        amp_sweeper = Sweeper(
            parameter=Parameter.amplitude,
            range=(
                params.amplitude_start,
                params.amplitude_stop,
                params.amplitude_step,
            ),
            pulses=[ro_pulses[qubit] for qubit in targets],
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
                        for f, ro_pulse in enumerate(readouts):
                            # l signal if it's the first or second measurment for QND
                            data.register_qubit(
                                ResonatorOptimizationType,
                                (qubit, state, f),
                                dict(
                                    frequency=np.array([freq]),
                                    amplitude=np.array([amp]),
                                    iq_values=np.array([results[ro_pulse.id][n][j][k]]),
                                ),
                            )
                            data.samples[qubit, state, f] = results_samples[
                                ro_pulse.id
                            ].tolist()

    return data


def _fit(data: ResonatorOptimizationData) -> ResonatorOptimizationResults:
    # qubits = data.qubits
    best_freq = {}
    best_amps = {}
    best_angle = {}
    best_threshold = {}
    highest_fidelity = {}

    # for qubit in qubits:
    ######################### ASSIGNMENT FIDELITY ##########################

    # for state, l in itertools.product([0, 1], [0, 1]):
    # freq_vals = np.unique(data[qubit, state, l]["frequency"])
    # amp_vals = np.unique(data[qubit, state, l]["amplitude"])
    # fidelity_grid = np.zeros(shape=(len(freq_vals), len(amp_vals)))
    # angle_grid = np.zeros(shape=(len(freq_vals), len(amp_vals)))
    # threshold_grid = np.zeros(shape=(len(freq_vals), len(amp_vals)))

    # for j, freq in enumerate(freq_vals):
    #    for k, amp in enumerate(amp_vals):

    ######################### QND ##########################################

    # TODO I have to computer the QND for each combined value of frequency and amplitude
    return ResonatorOptimizationResults(
        best_amp=best_amps,
        fidelities=highest_fidelity,
        best_freq=best_freq,
        best_angle=best_angle,
        best_threshold=best_threshold,
    )


def _plot(
    data: ResonatorOptimizationData, fit: ResonatorOptimizationResults, target: QubitId
):
    """Plotting function for resonator optimization"""

    figures = []
    fitting_report = ""

    fig = make_subplots(
        rows=1,
        cols=1,
    )

    qubit_data = data[target]
    frequencies = qubit_data.frequency
    amplitudes = qubit_data.amplitude
    fidelities = qubit_data.avaraged_fidelity

    if fit is not None:
        fig.add_trace(
            go.Heatmap(
                x=amplitudes,
                y=frequencies * HZ_TO_GHZ,
                z=fidelities,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[fit.best_amp],
                y=[fit.best_freq],
                mode="markers",
                marker=dict(
                    size=8,
                    color="black",
                    symbol="cross",
                ),
                name="highest assignment fidelity",
                showlegend=True,
            )
        )

        fitting_report = table_html(
            table_dict(
                target,
                [
                    "Best Readout Amplitude [a.u.]",
                    "Best Readout Frequency [GHz]",
                    "Best Fidelity",
                ],
                [
                    np.round(fit.best_amp[target], 4),
                    np.round(fit.best_freq[target]) * HZ_TO_GHZ,
                    fit.fidelities[target],
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
    update.readout_amplitude(results.best_amp[target], platform, target)
    update.readout_frequency(results.best_freq[target], platform, target)
    update.iq_angle(results.best_angle[target], platform, target)
    update.threshold(results.best_threshold[target], platform, target)


resonator_optimization = Routine(
    _acquisition,
    _fit,
    _plot,
    _update,
)
"""Resonator optimization Routine object"""
