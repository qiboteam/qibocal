from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, Parameter, Pulse, Sweeper

from qibocal.auto.operation import QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform

from ...config import log
from ..ramsey.utils import fitting, ramsey_sequence
from ..utils import COLORBAND, COLORBAND_LINE, HZ_TO_GHZ
from .t1_flux import T1FluxData, T1FluxParameters


@dataclass
class T2FluxParameters(T1FluxParameters):
    """T2 flux runcard inputs."""


@dataclass
class T2FluxResults(Results):
    """T2 flux outputs."""

    t2: dict[QubitId, list[float]] = field(default_factory=dict)
    """List of T2 value for each detuning value."""


@dataclass
class T2FluxData(T1FluxData):
    """T2 flux acquisition outputs."""


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

    sweeper_delay = Sweeper(
        parameter=Parameter.duration,
        values=params.delay_range,
        pulses=pulses,
    )

    sweeper_amplitude = Sweeper(
        parameter=Parameter.amplitude,
        values=params.flux_range,
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
        wait_range=params.delay_range.tolist(),
        flux_range=params.flux_range.tolist(),
        detuning={
            qubit: (
                platform.config(platform.qubits[qubit].drive).frequency * HZ_TO_GHZ
                + platform.calibration.single_qubits[qubit].qubit.detuning(
                    params.flux_range
                )
            ).tolist()
            for qubit in targets
        },
    )
    return data


def _fit(data: T2FluxData) -> T2FluxResults:
    """T2 flux fitting function.

    For each detuning value we compute the T2."""
    t2s = {qubit: [] for qubit in data.qubits}
    for qubit in data.qubits:
        prob = data.probability(qubit)
        error = data.error(qubit)
        for i in range(len(data.detuning[qubit])):
            try:
                popt, perr = fitting(np.array(data.wait_range), prob[i], error[i])
                t2 = 1 / popt[4]
                t2s[qubit].append([t2, perr[4] * (t2**2)])
            except Exception as e:
                log.warning(
                    f"T2 fitting failed for qubit {qubit} and amplitude {data.flux_range[i]} due to {e}."
                )
                t2s[qubit].append(np.nan)
    return T2FluxResults(t2=t2s)


def _plot(data: T2FluxData, target: QubitId, fit: T2FluxResults = None):
    """Plotting function for T2 experiment."""
    fig = go.Figure()
    if fit is not None:
        indices = list(set(np.where(np.array(fit.t2[target]) != np.nan)[0]))
        t2s = np.array([fit.t2[target][i][0] for i in indices])
        error = np.array([fit.t2[target][i][1] for i in indices])
        detuning = np.array(data.detuning[target])[indices]
        fig.add_traces(
            [
                go.Scatter(
                    x=detuning,
                    y=t2s,
                    opacity=1,
                    name="T2",
                    showlegend=True,
                    legendgroup="T2",
                    mode="lines",
                ),
                go.Scatter(
                    x=np.concatenate((detuning, detuning[::-1])),
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
            yaxis_title="T1 [ns]",
            yaxis=dict(range=[0, max(t2s) * 1.2]),
            xaxis=dict(range=[min(data.detuning[target]), max(data.detuning[target])]),
        )
    return [fig], ""


t2_flux = Routine(_acquisition, _fit, _plot)
"""T2 Routine object."""
