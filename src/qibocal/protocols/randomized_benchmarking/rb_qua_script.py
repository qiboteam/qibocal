from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.auto.execute import Executor
from qibocal.auto.history import History
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.cli.report import report
from qibocal.protocols import utils

RBCorrectionType = np.dtype(
    [
        ("biases", np.float64),
        ("qubit_frequency", np.float64),
        ("pulse_fidelity_uncorrected", np.float64),
        ("pulse_fidelity_corrected", np.float64),
    ]
)
"""Custom dtype for RBCorrection routines."""


@dataclass
class RBCorrectionSignalParameters(Parameters):
    """Coherence runcard inputs."""

    biases_start: list[float]
    biases_end: list[float]
    biases_step: list[float]

    # Flipping
    nflips_max: int
    """Maximum number of flips ([RX(pi) - RX(pi)] sequences). """
    nflips_step: int
    """Flip step."""

    # Ramsey signal
    delay_between_pulses_start: int
    """Initial delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_end: int
    """Final delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_step: int
    """Step delay between RX(pi/2) pulses in ns."""
    detuning: int
    """Frequency detuning [Hz]."""

    # Std Rb Ondevice Parameters
    num_of_sequences: int
    max_circuit_depth: int
    "Maximum circuit depth"
    delta_clifford: int
    "Play each sequence with a depth step equals to delta_clifford"
    seed: Optional[int] = None
    "Pseudo-random number generator seed"
    n_avg: int = 1
    "Number of averaging loops for each random sequence"
    save_sequences: bool = True
    apply_inverse: bool = True
    state_discrimination: bool = True
    "Flag to enable state discrimination if the readout has been calibrated (rotated blobs and threshold)"


@dataclass
class RBCorrectionSignalData(Data):
    """Coherence acquisition outputs."""

    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""

    @property
    def average(self):
        if len(next(iter(self.data.values())).shape) > 1:
            return utils.average_single_shots(self.__class__, self.data)
        return self


@dataclass
class RBCorrectionSignalResults(Results):
    sweetspot: dict[QubitId, float]
    "Best frequency for each qubit only regarding Coherence"


def _acquisition(
    params: RBCorrectionSignalParameters, platform: Platform, targets: list[QubitId]
) -> RBCorrectionSignalData:
    r"""Data acquisition for Coherence experiment."""

    executor = Executor.create(name="myexec", platform=platform)
    from myexec import (
        close,
        flipping_signal,
        init,
        ramsey_signal,
        rb_ondevice,
        single_shot_classification,
    )

    data = RBCorrectionSignalData()

    for target in targets:

        ss = platform.qubits[target].sweetspot
        # TODO: Center around the sweetspot and change arounf 10MHz
        biases = np.arange(
            params.biases_start + ss, params.biases_end + ss, params.biases_step
        )

        i = 0
        for bias in biases:
            i += 1
            # Change the flux
            platform.qubits[target].flux.offset = bias

            init(
                f"test_rb_correction/flux{i}", force=True, targets=targets
            )  # FIXME: Routine path

            rb_output_uncorrected = rb_ondevice(
                depths=params.depths,
                niter=params.niter,
                uncertainties=params.uncertainties,
                seed=params.seed,
                nshots=params.nshots,
            )

            ramsey_output = ramsey_signal(
                delay_between_pulses_start=params.delay_between_pulses_start,
                delay_between_pulses_end=params.delay_between_pulses_end,
                delay_between_pulses_step=params.delay_between_pulses_step,
                detuning=params.detuning,
            )
            ramsey_output.update_platform(platform)

            flipping_output = flipping_signal(
                nflips_max=params.nflips_max,
                nflips_step=params.nflips_step,
            )
            flipping_output.update_platform(platform)

            discrimination_output = single_shot_classification(nshots=5000)
            discrimination_output.update_platform(platform)

            rb_output_corrected = rb_ondevice(
                depths=params.depths,
                niter=params.niter,
                uncertainties=params.uncertainties,
                seed=params.seed,
                nshots=params.nshots,
            )

            data.register_qubit(
                RBCorrectionType,
                (target),
                dict(
                    biases=[bias],
                    qubit_frequency=[platform.qubits[target].native_gates.RX.frequency],
                    pulse_fidelity_uncorrected=[
                        rb_output_uncorrected.results.pulse_fidelity[target]
                    ],
                    pulse_fidelity_corrected=[
                        rb_output_corrected.results.pulse_fidelity[target]
                    ],
                ),
            )

            # close()
            report(executor.path, executor.history)
            executor.history = History()

    # stop and disconnect platform
    close()

    return data


def _fit(data: RBCorrectionSignalData) -> RBCorrectionSignalResults:
    """Post-processing function for 1FluxSignal."""

    return RBCorrectionSignalResults(sweetspot=None)


def _plot(data: RBCorrectionSignalData, target: QubitId, fit=None):
    """Plotting function for Coherence experiment."""

    figures = []
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=data[target].qubit_frequency,
            y=data[target].pulse_fidelity_uncorrected,
            opacity=1,
            name="Pulse Fidelity Uncorrected",
            showlegend=True,
            legendgroup="Pulse Fidelity Uncorrected",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=data[target].qubit_frequency,
            y=data[target].pulse_fidelity_corrected,
            opacity=1,
            name="Pulse Fidelity Corrected",
            showlegend=True,
            legendgroup="Pulse Fidelity Corrected",
        )
    )

    # last part
    figure.update_layout(
        showlegend=True,
        xaxis_title="Frequency [GHZ]",
        yaxis_title="Pulse Fidelity",
    )

    figures.append(figure)

    return figures, ""


def _update(results, platform: Platform, target: QubitId):
    pass


rb_correction_ondevice = Routine(_acquisition, _fit, _plot, _update)
"""RB Correction Routine object."""
