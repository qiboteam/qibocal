from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import zhinst.utils.shfqa
from laboneq.simple import pulse_library
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine


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
        ("msr", np.float64),
        ("phase", np.float64),
    ]
)
"""Custom dtype for CalibrateStateDiscrimination."""


@dataclass
class CalibrateStateDiscriminationData(Data):
    """CalibrateStateDiscrimination acquisition outputs."""

    resonator_type: str
    """Resonator type."""
    data: dict[
        tuple[QubitId, int], npt.NDArray[CalibrateStateDiscriminationType]
    ] = field(default_factory=dict)

    def register_qubit(self, qubit, state, msr, phase, i, q):
        """Store output for single qubit."""
        ar = np.empty(i.shape, dtype=CalibrateStateDiscriminationType)
        ar["msr"] = msr
        ar["phase"] = phase
        ar["i"] = i
        ar["q"] = q
        self.data[qubit, state] = np.rec.array(ar)


def _acquisition(
    params: CalibrateStateDiscriminationParameters, platform: Platform, qubits: Qubits
) -> CalibrateStateDiscriminationData:
    r"""
    Data acquisition for Calibrate State Discrimination experiment.
    Perform spectroscopy on the readout resonator, with the qubit in ground and excited state, showing
    the resonator shift produced by the coupling between the resonator and the qubit.

    Args:
        params (CalibrateStateDiscriminationParameters): experiment's parameters
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
            qubit, start=qd_pulses[qubit].duration
        )
        sequence_0.add(ro_pulses[qubit])
        sequence_1.add(qd_pulses[qubit])
        sequence_1.add(ro_pulses[qubit])

    # create a DataUnits objects to store the results
    data = CalibrateStateDiscriminationData(resonator_type=platform.resonator_type)

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

    # retrieve the results for every qubit
    for qubit in qubits:
        # average msr, phase, i and q over the number of shots defined in the runcard
        for i, results in enumerate([results_0, results_1]):
            result = results[ro_pulses[qubit].serial]
            # store the results
            data.register_qubit(
                qubit=qubit,
                state=i,
                msr=result.magnitude,
                phase=result.phase,
                i=result.voltage_i,
                q=result.voltage_q,
            )
    return data


def _fit(data: CalibrateStateDiscriminationData) -> CalibrateStateDiscriminationResults:
    """Post-Processing for Calibrate State Discrimination"""
    qubits = data.qubits

    kernel_state_zero = {}
    for qubit in qubits:
        traces = []
        for i in range(2):
            # TODO: Are traces calculated with these values or MSR or Phase???
            trace = (
                data[qubit, i]["i"][: (len(data[qubit, i]["i"]) // 16) * 16]
                + 1j * data[qubit, i]["q"][: (len(data[qubit, i]["i"]) // 16) * 16]
            )
            traces.append(trace)

        qdit_settings = zhinst.utils.shfqa.multistate.QuditSettings(ref_traces=traces)
        num_states = 2
        weights = qdit_settings.weights[: num_states - 1]
        kernels = [pulse_library.sampled_pulse_complex(w.vector) for w in weights]
        kernel_state_zero[qubit] = kernels[0].samples

    return CalibrateStateDiscriminationResults(data=kernel_state_zero)


def _plot(
    data: CalibrateStateDiscriminationData,
    qubit,
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
                x=fit.data[qubit].real,
                y=fit.data[qubit].imag,
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
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
            xaxis_title="Kernel Imag",
            yaxis_title="Kernel Real",
        )

        figures.append(fig)

    return figures, fitting_report


def _update(
    results: CalibrateStateDiscriminationResults, platform: Platform, qubit: QubitId
):
    pass


calibrate_state_discrimination = Routine(_acquisition, _fit, _plot, _update)
"""Calibrate State Discrimination Routine object."""
