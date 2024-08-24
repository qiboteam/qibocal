from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Results, Routine

SAMPLES_FACTOR = 16


@dataclass
class CalibrateStateDiscriminationParameters(Parameters):
    """Calibrate State Discrimination inputs."""

    """Frequency step for sweep (Hz)."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


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
        ("signal", np.float64),
        ("phase", np.float64),
    ]
)
"""Custom dtype for CalibrateStateDiscrimination."""


@dataclass
class CalibrateStateDiscriminationData(Data):
    """CalibrateStateDiscrimination acquisition outputs."""

    resonator_type: str
    """Resonator type."""
    data: dict[tuple[QubitId, int], npt.NDArray[CalibrateStateDiscriminationType]] = (
        field(default_factory=dict)
    )


def _acquisition(
    params: CalibrateStateDiscriminationParameters,
    platform: Platform,
    targets: list[QubitId],
) -> CalibrateStateDiscriminationData:
    r"""
    Data acquisition for Calibrate State Discrimination experiment.
    Calculates the optimal kernel for the readout. It has to be run one qubit at a time.
    The kernels are stored in the result.npz generated on the report.

    Args:
        params (CalibrateStateDiscriminationParameters): experiment's parameters
        platform (Platform): Qibolab platform object
        qubits (dict): list of target qubits to perform the action

    """

    # create 2 sequences of pulses for the experiment:
    # sequence_0: I  - MZ
    # sequence_1: RX - MZ

    data = CalibrateStateDiscriminationData(resonator_type=platform.resonator_type)

    # TODO: test if qibolab supports multiplex with raw acquisition
    for qubit in targets:
        sequence_0 = PulseSequence()
        sequence_1 = PulseSequence()
        sequence_1.add(platform.create_RX_pulse(qubit, start=0))

        sequence_0.add(
            platform.create_qubit_readout_pulse(qubit, start=sequence_0.finish)
        )
        sequence_1.add(
            platform.create_qubit_readout_pulse(qubit, start=sequence_1.finish)
        )

        results_0 = platform.execute_pulse_sequence(
            sequence_0,
            ExecutionParameters(
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.RAW,
                averaging_mode=AveragingMode.CYCLIC,
            ),
        )

        results_1 = platform.execute_pulse_sequence(
            sequence_1,
            ExecutionParameters(
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.RAW,
                averaging_mode=AveragingMode.CYCLIC,
            ),
        )

        for i, experiment in enumerate(
            zip([sequence_0, sequence_1], [results_0, results_1])
        ):
            sequence, results = experiment
            result = results[sequence.ro_pulses[0].serial]
            # store the results
            data.register_qubit(
                CalibrateStateDiscriminationType,
                (qubit, i),
                dict(
                    signal=result.magnitude,
                    phase=result.phase,
                    i=result.voltage_i,
                    q=result.voltage_q,
                ),
            )

    return data


def _fit(data: CalibrateStateDiscriminationData) -> CalibrateStateDiscriminationResults:
    """Post-Processing for Calibrate State Discrimination"""
    qubits = data.qubits

    kernel_state_zero = {}
    for qubit in qubits:
        traces = []
        for i in range(2):
            # This is due to a device limitation on the number of samples
            # We want the number of samples to be a factor of 16
            trace = (
                data[qubit, i]["i"][
                    : (len(data[qubit, i]["i"]) // SAMPLES_FACTOR) * SAMPLES_FACTOR
                ]
                + 1j
                * data[qubit, i]["q"][
                    : (len(data[qubit, i]["i"]) // SAMPLES_FACTOR) * SAMPLES_FACTOR
                ]
            )
            traces.append(trace)
        """
        This is a simplified version from laboneq.analysis.calculate_integration_kernels
        for our use case where we only want to discirminate between state 0 and 1
        """
        # Calculate the optimal kernel
        kernel = np.conj(traces[0] - traces[1])

        # Normalize the kernel
        max_abs_weight = max(np.abs(kernel))
        kernel *= 1 / max_abs_weight

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

    if fit is not None:
        fig = make_subplots(
            rows=1,
            cols=1,
            horizontal_spacing=0.1,
            vertical_spacing=0.1,
            subplot_titles=("Kernel state 0",),
        )

        fig.add_trace(
            go.Scatter(
                x=np.arange(len(fit.data[target])),
                y=np.abs(fit.data[target]),
                opacity=1,
                name="kernel state 0",
                showlegend=True,
                legendgroup="kernel state 0",
            ),
            row=1,
            col=1,
        )

        fig.update_layout(
            showlegend=True,
            xaxis_title="Kernel samples",
            yaxis_title="Kernel absolute value",
        )

        figures.append(fig)

    return figures, fitting_report


def _update(
    results: CalibrateStateDiscriminationResults, platform: Platform, qubit: QubitId
):
    update.kernel(results.data[qubit], platform, qubit)


calibrate_state_discrimination = Routine(_acquisition, _fit, _plot, _update)
"""Calibrate State Discrimination Routine object."""
