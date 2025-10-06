from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, PulseSequence
from scipy.signal import butter, filtfilt

from qibocal import update
from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform

from .utils import _get_lo_frequency

__all__ = ["calibrate_state_discrimination"]


@dataclass
class CalibrateStateDiscriminationParameters(Parameters):
    """Calibrate State Discrimination inputs."""

    """Frequency step for sweep (Hz)."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""
    unrolling: Optional[bool] = False


CalibrateStateDiscriminationResType = np.dtype(
    [
        ("State 0 kernel", np.float64),
    ]
)
"""Custom dtype for CalibrateStateDiscrimination."""


@dataclass
class CalibrateStateDiscriminationResults(Results):
    """Calibrate State Discrimination outputs."""

    data: dict[
        tuple[QubitId, int], npt.NDArray[CalibrateStateDiscriminationResType]
    ] = field(default_factory=dict)
    """State 0 kernel"""


CalibrateStateDiscriminationType = np.dtype(
    [
        ("i", np.float64),
        ("q", np.float64),
    ]
)
"""Custom dtype for CalibrateStateDiscrimination."""


@dataclass
class CalibrateStateDiscriminationData(Data):
    """CalibrateStateDiscrimination acquisition outputs."""

    resonator_type: str
    """Resonator type."""
    intermediate_frequency: dict[QubitId, float] = field(default_factory=dict)
    """Intermediate RO frequency for each qubit."""
    data: dict[tuple[QubitId, int], npt.NDArray[CalibrateStateDiscriminationType]] = (
        field(default_factory=dict)
    )


def _acquisition(
    params: CalibrateStateDiscriminationParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> CalibrateStateDiscriminationData:
    r"""
    Data acquisition for Calibrate State Discrimination experiment.
    Calculates the optimal kernel for the readout. It has to be run one qubit at a time.
    The kernels are stored in the result.npz generated on the report.

    Args:
        params (CalibrateStateDiscriminationParameters): experiment's parameters
        platform (CalibrationPlatform): Qibolab platform object
        qubits (dict): list of target qubits to perform the action

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

    options = dict(
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.RAW,
        averaging_mode=AveragingMode.CYCLIC,
    )

    if params.unrolling:
        results = platform.execute(sequences, **options)
    else:
        results = {}
        for sequence in sequences:
            results.update(platform.execute([sequence], **options))

    intermediate_frequency = {
        qubit: platform.config(platform.qubits[qubit].probe).frequency
        - _get_lo_frequency(platform, qubit)
        for qubit in targets
    }
    data = CalibrateStateDiscriminationData(
        resonator_type=platform.resonator_type,
        intermediate_frequency=intermediate_frequency,
    )
    for state, ro_pulses in zip([0, 1], all_ro_pulses):
        for qubit in targets:
            serial = ro_pulses[qubit]
            result = results[serial]
            data.register_qubit(
                CalibrateStateDiscriminationType,
                (qubit, state),
                dict(
                    i=result[..., 0],
                    q=result[..., 1],
                ),
            )
    return data


def _fit(data: CalibrateStateDiscriminationData) -> CalibrateStateDiscriminationResults:
    """Post-Processing for Calibrate State Discrimination.

    The raw traces for 0 and 1 are demodulated with the IF stored during acquisition.
    Moreover, fast oscillating terms (:math:`\\exp(-2\\pi \\omega_{\text{IF} t)`) are removed by applying a lowpass filter.
    Finally, the optimal kernel is calculated as the difference between the two traces.
    """
    qubits = data.qubits

    kernel_state_zero = {}

    def lowpass_filter(signal, cutoff, fs=1e9, order=5):
        """Basic lowpass filter."""
        norm_cutoff = cutoff / fs
        b, a = butter(order, norm_cutoff)
        return filtfilt(b, a, signal)

    for qubit in qubits:
        traces = []
        freq = data.intermediate_frequency[qubit]

        for i in range(2):
            trace = data[qubit, i].i + 1.0j * data[qubit, i].q
            t = np.arange(0, len(trace), 1)

            # demodulation (we assume that RAW doesn't demodulate)
            trace = np.array(
                [
                    np.exp(-2 * np.pi * t[i] * 1.0j * freq * 1e-9) * trace[i]
                    for i in range(len(t))
                ]
            )
            # apply lowpass filter to remove fast rotating terms
            trace = lowpass_filter(trace, 2 * np.abs(freq))
            traces.append(trace)

        # Calculate the optimal kernel
        kernel = np.conj(traces[0] - traces[1])

        # Normalize the kernel
        norm = np.sqrt(np.sum(np.abs(kernel) ** 2))
        max_abs_weight = np.max(np.abs(kernel / norm))
        kernel /= norm * max_abs_weight
        kernel_state_zero[qubit] = kernel

    return CalibrateStateDiscriminationResults(data=kernel_state_zero)


def _plot(
    data: CalibrateStateDiscriminationData,
    target,
    fit: CalibrateStateDiscriminationResults,
):
    """Plotting function for Calibrate State Discrimination."""
    # Plot kernels
    figures = []
    fitting_report = ""
    trace0 = data[target, 0].i + 1.0j * data[target, 0].q
    t = np.arange(0, len(trace0), 1)

    if fit is not None:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=t,
                y=fit.data[target].real,
                opacity=1,
                name="Real",
                showlegend=True,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=t,
                y=fit.data[target].imag,
                opacity=1,
                name="Imag",
                showlegend=True,
            )
        )

        fig.update_layout(
            showlegend=True,
            title="Optimal integration kernel",
            xaxis_title="Readout sample",
            yaxis_title="Normalized weight",
        )

        figures.append(fig)

    return figures, fitting_report


def _update(
    results: CalibrateStateDiscriminationResults,
    platform: CalibrationPlatform,
    qubit: QubitId,
):
    update.kernel(results.data[qubit], platform, qubit)


calibrate_state_discrimination = Routine(_acquisition, _fit, _plot, _update)
"""Calibrate State Discrimination Routine object."""
