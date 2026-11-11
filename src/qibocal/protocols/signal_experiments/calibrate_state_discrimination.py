from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, PulseSequence
from scipy.signal import butter, sosfiltfilt

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Protocol, QubitId, Results
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import table_dict, table_html

from .utils import _get_lo_frequency

__all__ = ["calibrate_state_discrimination"]


@dataclass
class CalibrateStateDiscriminationParameters(Parameters):
    """Calibrate State Discrimination inputs."""

    lowpass_cutoff: float | None = None
    """Low-pass cutoff (Hz) for the traces.

    Choose a few resonator linewidths. If ``None``, the intermediate frequency is used.
    """


@dataclass
class CalibrateStateDiscriminationResults(Results):
    """Calibrate State Discrimination outputs."""

    data: dict[QubitId, npt.NDArray[np.complex128]] = field(default_factory=dict)
    """Integration kernel of each qubit."""


@dataclass
class CalibrateStateDiscriminationData(Data):
    """CalibrateStateDiscrimination acquisition outputs."""

    lowpass_cutoff: float | None
    """Low-pass cutoff (Hz) for the traces."""
    intermediate_frequency: dict[QubitId, float] = field(default_factory=dict)
    """Intermediate RO frequency for each qubit."""
    data: dict[tuple[QubitId, int], npt.NDArray[np.float64]] = field(
        default_factory=dict
    )
    """Raw traces of each qubit and state, with I and Q on the last axis."""


def _acquisition(
    params: CalibrateStateDiscriminationParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> CalibrateStateDiscriminationData:
    """Data acquisition for Calibrate State Discrimination experiment.

    Acquires the RAW readout traces with the qubits prepared in the ground and excited
    state, from which the optimal kernels are calculated.

    Args:
        params (CalibrateStateDiscriminationParameters): experiment's parameters
        platform (CalibrationPlatform): Qibolab platform object
        targets (list): list of target qubits to perform the action
    """

    native = platform.natives.single_qubit
    sequences, all_ro_pulses = [], []
    for state in [0, 1]:
        ro_pulses = {}
        sequence = PulseSequence()
        for q in targets:
            ro_sequence = native[q].MZ()
            ro_pulses[q] = ro_sequence[0][1].id
            sequence += ro_sequence

        if state == 1:
            rx_sequence = PulseSequence()
            for q in targets:
                rx_sequence += native[q].RX()
            sequence = rx_sequence | sequence

        sequences.append(sequence)
        all_ro_pulses.append(ro_pulses)

    options = {
        "nshots": params.nshots,
        "relaxation_time": params.relaxation_time,
        "acquisition_type": AcquisitionType.RAW,
        "averaging_mode": AveragingMode.CYCLIC,
    }

    results = platform.execute(sequences, **options)

    intermediate_frequency = {
        qubit: platform.config(platform.qubits[qubit].probe).frequency
        - _get_lo_frequency(platform, qubit)
        for qubit in targets
    }
    data = CalibrateStateDiscriminationData(
        lowpass_cutoff=params.lowpass_cutoff,
        intermediate_frequency=intermediate_frequency,
    )
    for state, ro_pulses in zip([0, 1], all_ro_pulses):
        for qubit in targets:
            data.data[qubit, state] = results[ro_pulses[qubit]]
    return data


def _fit(data: CalibrateStateDiscriminationData) -> CalibrateStateDiscriminationResults:
    r"""Post-Processing for Calibrate State Discrimination.

    The raw traces for 0 and 1 are already demodulated to baseband by Qibolab. A lowpass
    filter at ``lowpass_cutoff`` removes the averaging noise above the readout signal
    bandwidth. If ``lowpass_cutoff`` is ``None``, the cutoff is set to the IF. The
    cutoff never exceeds :math:`|f_{\text{IF}}|`, halfway between baseband and the
    residual image (:math:`\exp(-2\pi i \cdot 2 f_{\text{IF}} t)`), which is then
    suppressed as well. The optimal kernel is calculated as the conjugate of the
    difference between the two traces, and it is normalized to a maximum absolute weight
    of 1.
    """
    qubits = data.qubits

    kernels = {}

    def lowpass_filter(signal, cutoff, fs=1e9, order=5):
        """Basic lowpass filter, with ``cutoff`` and sampling rate ``fs`` in Hz."""
        sos = butter(order, cutoff, fs=fs, output="sos")
        return sosfiltfilt(sos, signal)

    for qubit in qubits:
        traces = []
        freq = abs(data.intermediate_frequency[qubit])
        # remove averaging noise above a few resonator linewidths, and the 2 * IF image,
        # by never exceeding the IF
        cutoff = data.lowpass_cutoff
        if freq:
            cutoff = freq if cutoff is None else min(cutoff, freq)

        for state in range(2):
            trace = data[qubit, state] @ [1, 1j]
            traces.append(trace if cutoff is None else lowpass_filter(trace, cutoff))

        # Calculate the optimal kernel
        kernel = np.conj(traces[0] - traces[1])
        kernel /= np.max(np.abs(kernel))
        kernels[qubit] = kernel

    return CalibrateStateDiscriminationResults(data=kernels)


def _plot(
    data: CalibrateStateDiscriminationData,
    target,
    fit: CalibrateStateDiscriminationResults,
):
    """Plotting function for Calibrate State Discrimination."""
    # Plot kernels
    figures = []
    fitting_report = ""
    t = np.arange(len(data[target, 0]))

    fig = go.Figure()
    for path, name in enumerate("IQ"):
        for state in range(2):
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=data[target, state][:, path],
                    name=f"State {state} {name}",
                    mode="lines",
                )
            )
    fig.update_layout(
        title="Readout traces",
        xaxis_title="Time (ns)",
        yaxis_title="Signal [a.u.]",
    )
    figures.append(fig)

    if fit is not None:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=t,
                y=fit.data[target].real,
                opacity=1,
                name="Real",
                showlegend=True,
                mode="lines",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=t,
                y=fit.data[target].imag,
                opacity=1,
                name="Imaginary",
                showlegend=True,
                mode="lines",
            )
        )

        fig.update_layout(
            showlegend=True,
            title="Optimal integration kernel",
            xaxis_title="Time (ns)",
            yaxis_title="Normalized weight",
        )

        figures.append(fig)

        # the kernel is the conjugate of the separation: compare its SNR with the one
        # of a square window, which weights all the samples equally
        kernel = fit.data[target]
        gain = np.sqrt(
            len(kernel) * np.sum(np.abs(kernel) ** 2) / np.abs(np.sum(kernel)) ** 2
        )
        fitting_report = table_html(
            table_dict(
                target,
                ["Estimated SNR gain over square window"],
                [round(float(gain), 3)],
            )
        )

    return figures, fitting_report


def _update(
    results: CalibrateStateDiscriminationResults,
    platform: CalibrationPlatform,
    qubit: QubitId,
):
    update.kernel(results.data[qubit], platform, qubit)


calibrate_state_discrimination = Protocol(_acquisition, _fit, _plot, _update)
"""Calibrate State Discrimination Protocol object."""
