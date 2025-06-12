from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, Parameter, Pulse, Sweeper

from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.result import probability

from ..ramsey.utils import fitting, ramsey_sequence
from ..utils import COLORBAND, COLORBAND_LINE, HZ_TO_GHZ


@dataclass
class T2FluxParameters(Parameters):
    """T2 flux runcard inputs."""

    delay_between_pulses_start: int
    """Initial delay before readout [ns]."""
    delay_between_pulses_end: int
    """Final delay before readout [ns]."""
    delay_between_pulses_step: int
    """Step delay before readout [ns]."""
    amplitude_min: float
    """Flux pulse minimum amplitude."""
    amplitude_max: float
    """Flux pulse maximum amplitude."""
    amplitude_step: float
    """Flux pulse amplitude step."""


@dataclass
class T2FluxResults(Results):
    """T2 flux outputs."""

    t2: dict[QubitId, list[float]] = field(default_factory=dict)
    """List of T2 value for each detuning value."""


@dataclass
class T2FluxData(Data):
    """T2 flux acquisition outputs."""

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
    params: T2FluxParameters, platform: CalibrationPlatform, targets: list[QubitId]
) -> T2FluxData:
    """Data acquisition for T2 flux experiment."""

    sequence, pulses = ramsey_sequence(
        platform=platform, targets=targets, flux_pulse_amplitude=0.5
    )
    for qubit in targets:
        assert (
            platform.calibration.single_qubits[qubit].qubit.flux_coefficients
            is not None
        ), f"Qubit {qubit} flux coefficients not set in calibration."
    wait_range = np.arange(
        params.delay_between_pulses_start,
        params.delay_between_pulses_end,
        params.delay_between_pulses_step,
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
        ro_pulse = list(sequence.channel(platform.qubits[qubit].acquisition))[-1]
        _data[qubit] = results[ro_pulse.id]
    data = T2FluxData(
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


def _fit(data: T2FluxData) -> T2FluxResults:
    t2s = {qubit: [] for qubit in data.qubits}
    for qubit in data.qubits:
        prob = data.probability(qubit)
        error = data.error(qubit)
        for i in range(len(data.detuning[qubit])):
            popt, perr = fitting(np.array(data.wait_range), prob[i], error[i])
            t2 = 1 / popt[4]
            t2s[qubit].append([t2, perr[4] * (t2**2)])
    return T2FluxResults(t2=t2s)


def _plot(data: T2FluxData, target: QubitId, fit: T2FluxResults = None):
    """Plotting function for T2 experiment."""
    fig = go.Figure()
    if fit is not None:
        t2s = np.array([fit.t2[target][i][0] for i in range(len(fit.t2[target]))])
        error = np.array([fit.t2[target][i][1] for i in range(len(fit.t2[target]))])
        fig.add_traces(
            [
                go.Scatter(
                    x=data.detuning[target],
                    y=t2s,
                    opacity=1,
                    name="T2",
                    showlegend=True,
                    legendgroup="T2",
                    mode="lines",
                ),
                go.Scatter(
                    x=np.concatenate(
                        (data.detuning[target], data.detuning[target][::-1])
                    ),
                    y=np.concatenate((t2s + error, (t2s - error)[::-1])),
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
        yaxis_title="T2 [ns]",
    )
    return [fig], ""


t2_flux = Routine(_acquisition, _fit, _plot)
"""T2 Routine object."""
