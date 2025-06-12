from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, Parameter, Pulse, Sweeper

from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.result import probability

from ..utils import COLORBAND, COLORBAND_LINE, HZ_TO_GHZ
from .t1_signal import t1_sequence
from .utils import single_exponential_fit


@dataclass
class T1FluxParameters(Parameters):
    """T1 runcard inputs."""

    delay_before_readout_start: int
    """Initial delay before readout [ns]."""
    delay_before_readout_end: int
    """Final delay before readout [ns]."""
    delay_before_readout_step: int
    """Step delay before readout [ns]."""
    amplitude_min: float
    """Flux pulse minimum amplitude."""
    amplitude_max: float
    """Flux pulse maximum amplitude."""
    amplitude_step: float
    """Flux pulse amplitude step."""


@dataclass
class T1FluxResults(Results):
    """T1 outputs."""

    t1: dict[QubitId, list[float]] = field(default_factory=dict)


@dataclass
class T1FluxData(Data):
    """T1 acquisition outputs."""

    flux_range: list[float] = field(default_factory=list)
    """Flux pulse amplitude range [a.u.]."""
    wait_range: list[float] = field(default_factory=list)
    """Delay between pulses range [ns]."""
    detuning: dict[QubitId, float] = field(default_factory=dict)
    """DETUNING of the qubit as a function of flux pulse amplitude."""
    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""

    def probability(self, qubit: QubitId) -> npt.NDArray:
        """Return the probability data for a specific qubit."""
        return probability(self.data[qubit], state=1)

    def error(self, qubit: QubitId) -> npt.NDArray:
        """Return the error data for a specific qubit."""
        probs = self.probability(qubit)
        nshots = self.data[qubit].shape[0]
        return np.sqrt(probs * (1 - probs) / nshots)


def _acquisition(
    params: T1FluxParameters, platform: CalibrationPlatform, targets: list[QubitId]
) -> T1FluxData:
    """Data acquisition for T1 flux experiment."""

    sequence, ro_pulses, pulses = t1_sequence(
        platform=platform, targets=targets, flux_pulse_amplitude=0.5
    )
    for qubit in targets:
        assert (
            platform.calibration.single_qubits[qubit].qubit.flux_coefficients
            is not None
        ), f"Qubit {qubit} flux coefficients not set in calibration."
    wait_range = np.arange(
        params.delay_before_readout_start,
        params.delay_before_readout_end,
        params.delay_before_readout_step,
    )

    flux_range = np.arange(
        params.amplitude_min,
        params.amplitude_max,
        params.amplitude_step,
    )

    sweeper_delay = Sweeper(
        parameter=Parameter.duration,
        values=wait_range,
        pulses=pulses,
    )

    sweeper_amplitude = Sweeper(
        parameter=Parameter.amplitude,
        values=flux_range,
        pulses=[pulse for pulse in pulses if isinstance(pulse, Pulse)],
    )

    _data = {}

    results = platform.execute(
        [sequence],
        [[sweeper_amplitude], [sweeper_delay]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.SINGLESHOT,
    )

    for qubit in targets:
        _data[qubit] = results[ro_pulses[qubit].id]
    data = T1FluxData(
        data=_data,
        wait_range=wait_range.tolist(),
        flux_range=flux_range.tolist(),
        detuning={
            qubit: (
                platform.config(platform.qubits[qubit].drive).frequency * HZ_TO_GHZ
                + platform.calibration.single_qubits[qubit].qubit.detuning(flux_range)
            ).tolist()
            for qubit in targets
        },
    )
    return data


def _fit(data: T1FluxData) -> T1FluxResults:
    t1s = {qubit: [] for qubit in data.qubits}
    for qubit in data.qubits:
        prob = data.probability(qubit)
        error = data.error(qubit)
        for i in range(len(data.detuning[qubit])):
            t1, _, _, _ = single_exponential_fit(
                np.array(data.wait_range), prob[i], error[i], zeno=False
            )
            t1s[qubit].append(t1)
    return T1FluxResults(t1=t1s)


def _plot(data: T1FluxData, target: QubitId, fit: T1FluxResults = None):
    """Plotting function for T1 flux experiment."""
    fig = go.Figure()
    if fit is not None:
        t1s = np.array([fit.t1[target][i][0] for i in range(len(fit.t2[target]))])
        error = np.array([fit.t1[target][i][1] for i in range(len(fit.t2[target]))])
        fig.add_traces(
            [
                go.Scatter(
                    x=data.detuning[target],
                    y=t1s,
                    opacity=1,
                    name="T1",
                    showlegend=True,
                    legendgroup="T1",
                    mode="lines",
                ),
                go.Scatter(
                    x=np.concatenate(
                        (data.detuning[target], data.detuning[target][::-1])
                    ),
                    y=np.concatenate((t1s + error, (t1s - error)[::-1])),
                    fill="toself",
                    fillcolor=COLORBAND,
                    line=dict(color=COLORBAND_LINE),
                    showlegend=True,
                    name="Errors",
                ),
            ]
        )
    fig.update_layout(
        xaxis_title="Frequency [GHz]",
        yaxis_title="T1 [ns]",
    )
    return [fig], ""


t1_flux = Routine(_acquisition, _fit, _plot)
"""T1 flux Routine object."""
