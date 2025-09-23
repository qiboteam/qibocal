from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    Pulse,
    PulseSequence,
    Rectangular,
    Sweeper,
)
from scipy.constants import kilo
from scipy.signal import lfilter

from ... import update
from ...auto.operation import Data, Parameters, QubitId, Results, Routine
from ...calibration import CalibrationPlatform
from ...config import log
from ...result import magnitude
from ..utils import (
    GHZ_TO_HZ,
    HZ_TO_GHZ,
    extract_feature,
    readout_frequency,
    table_dict,
    table_html,
)
from .cryoscope import exponential_params, filter_calc

__all__ = ["long_cryoscope"]


@dataclass
class LongCryoscopeParameters(Parameters):
    """LongCryoscope runcard inputs."""

    duration_min: float
    """Minimum duration of delay between flux and drive pulse."""
    duration_max: float
    """Maximum duration of delay between flux and drive pulse."""
    duration_step: float
    """Step duration of delay between flux and drive pulse."""
    flux_pulse_amplitude: float
    """Flux pulse amplitude used for the sequence."""
    freq_width: float
    """Frequency width for drive frequency sweeper."""
    freq_step: float
    """Frequency step for drive frequency sweeper."""

    @property
    def frequency_range(self) -> np.ndarray:
        """Frequency range based on runcard parameters."""
        return np.arange(-self.freq_width / 2, self.freq_width / 2, self.freq_step)

    @property
    def duration_range(self) -> np.ndarray:
        """Duration range based on runcard parameters."""
        return np.arange(self.duration_min, self.duration_max, self.duration_step)


@dataclass
class LongCryoscopeResults(Results):
    """LongCryoscope outputs."""

    g: dict[QubitId, float] = field(default_factory=dict)
    exp_amplitude: dict[QubitId, list[float]] = field(default_factory=dict)
    """A parameters for the exp decay approximation"""
    tau: dict[QubitId, list[float]] = field(default_factory=dict)
    """time decay constant in exp decay approximation"""
    feedforward_taps: dict[QubitId, list[float]] = field(default_factory=dict)
    """feedforward taps"""
    feedback_taps: dict[QubitId, list[float]] = field(default_factory=dict)
    """feedback taps"""


@dataclass
class LongCryoscopeData(Data):
    """LongCryoscope acquisition outputs."""

    frequency: dict[QubitId, float]
    """Initial frequency for each qubit."""
    flux_coefficients: dict[QubitId, list]
    """Flux coefficients for each Qubit."""
    frequency_swept: dict[QubitId, list]
    """Exact frequencies swept for each qubit."""
    duration_swept: list
    """Duration swept list."""
    data: dict[QubitId, np.ndarray] = field(default_factory=dict)
    """Raw data."""

    def grid(self, qubit: QubitId) -> tuple[np.ndarray]:
        """Ravelling grid data."""
        x, y = np.meshgrid(self.frequency_swept[qubit], self.duration_swept)
        return x.ravel(), y.ravel(), magnitude(self.data[qubit]).ravel()

    def filtered_data(self, qubit: QubitId) -> tuple[np.ndarray]:
        """Extract relevant x and y."""
        freq, delay = extract_feature(*self.grid(qubit), find_min=False)
        return delay, freq

    def step_reponse(self, qubit: QubitId) -> np.ndarray:
        """Compute expected frequency by averaging over last half."""

        _, freq = self.filtered_data(qubit)
        freq_ghz = (freq - self.frequency[qubit]) * HZ_TO_GHZ
        p = np.poly1d(self.flux_coefficients[qubit])
        amplitude = [max((p - f).roots) for f in freq_ghz]
        return amplitude / np.mean(amplitude[len(freq_ghz) // 2 :])


def sequence(
    platform: CalibrationPlatform,
    target: QubitId,
    flux_pulse_amplitude: float,
    delay: float,
) -> PulseSequence:
    """Sequence used in the experiment for single qubit."""
    seq = PulseSequence()
    natives = platform.natives.single_qubit[target]
    qd_channel, qd_pulse = natives.RX()[0]
    ro_channel, ro_pulse = natives.MZ()[0]
    flux_channel = platform.qubits[target].flux
    flux_pulse = Pulse(
        duration=2 * delay + qd_pulse.duration,
        amplitude=flux_pulse_amplitude,
        envelope=Rectangular(),
    )
    seq.append((flux_channel, flux_pulse))
    seq.append((qd_channel, Delay(duration=delay)))
    seq.append((qd_channel, qd_pulse))
    seq.append((ro_channel, Delay(duration=flux_pulse.duration)))
    seq.append((ro_channel, ro_pulse))
    return seq


def _acquisition(
    params: LongCryoscopeParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> LongCryoscopeData:
    """Data acquisition for LongCryoscope Experiment."""

    freq_sweepers = []
    data_ = {q: [] for q in targets}
    for delay in params.duration_range:
        seq = PulseSequence()
        for q in targets:
            seq += sequence(platform, q, params.flux_pulse_amplitude, delay)
            qd_channel = platform.qubits[q].drive
            freq_sweepers.append(
                Sweeper(
                    parameter=Parameter.frequency,
                    values=platform.config(qd_channel).frequency
                    + platform.calibration.single_qubits[q].qubit.detuning(
                        params.flux_pulse_amplitude
                    )
                    * GHZ_TO_HZ
                    + params.frequency_range,
                    channels=[qd_channel],
                )
            )

        results = platform.execute(
            [seq],
            [freq_sweepers],
            updates=[
                {
                    platform.qubits[q].probe: {
                        "frequency": readout_frequency(q, platform)
                    }
                }
                for q in targets
            ],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        )

        for qubit in targets:
            acq_handle = list(seq.channel(platform.qubits[qubit].acquisition))[-1].id
            data_[qubit].append(results[acq_handle])

    data = LongCryoscopeData(
        frequency={
            qubit: platform.config(platform.qubits[qubit].drive).frequency
            for qubit in targets
        },
        flux_coefficients={
            qubit: platform.calibration.single_qubits[qubit].qubit.flux_coefficients
            for qubit in targets
        },
        frequency_swept={
            qubit: freq_sweepers[i].values.tolist() for i, qubit in enumerate(targets)
        },
        duration_swept=params.duration_range.tolist(),
        data={qubit: np.stack(result) for qubit, result in data_.items()},
    )
    return data


def exp_fit(x, tau, a, g):
    return g * (1 + a * np.exp(-x / tau))


def _fit(data: LongCryoscopeData) -> LongCryoscopeResults:
    """Postprocessing for long cryoscope experiment.

    An exponential fit is performed on the relevant points.

    """
    feedback_taps = {}
    feedforward_taps = {}
    time_decay = {}
    alpha = {}
    g = {}
    sampling_rate = 1 / (data.duration_swept[1] - data.duration_swept[0])
    for qubit in data.qubits:
        delay, _ = data.filtered_data(qubit)

        step_response = data.step_reponse(qubit)
        try:
            exp_params = exponential_params(delay, step_response)
            feedback_taps[qubit], feedforward_taps[qubit] = filter_calc(
                exp_params, sampling_rate
            )
            time_decay[qubit], alpha[qubit], g[qubit] = exp_params
        except RuntimeError:
            log.info("Fit failed")
    return LongCryoscopeResults(
        exp_amplitude=alpha,
        g=g,
        tau=time_decay,
        feedback_taps=feedback_taps,
        feedforward_taps=feedforward_taps,
    )


def _plot(data: LongCryoscopeData, fit: LongCryoscopeResults, target: QubitId):
    """Plotting function for LongCryoscope Experiment."""
    print(fit)
    fig = make_subplots(
        rows=2,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
        shared_xaxes=True,
    )
    fitting_report = ""
    delay, freq = data.filtered_data(target)
    step_response = data.step_reponse(target)
    fig.add_trace(
        go.Heatmap(
            x=data.duration_swept,
            y=np.array(data.frequency_swept[target]) * HZ_TO_GHZ,
            z=magnitude(data.data[target]).T,
            colorbar=dict(title="Signal [a.u.]"),
            colorscale="Viridis",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=delay,
            y=freq * HZ_TO_GHZ,
            mode="markers",
            showlegend=True,
            legendgroup="Data",
            name="Extract feature",
            marker=dict(color="rgb(248, 248, 248)"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=delay,
            y=step_response,
            showlegend=False,
            legendgroup="Data",
            mode="markers",
            name="Extract feature",
        ),
        row=2,
        col=1,
    )
    if fit is not None:
        fig.add_trace(
            go.Scatter(
                x=delay,
                y=exp_fit(
                    delay, fit.tau[target], fit.exp_amplitude[target], fit.g[target]
                ),
                showlegend=True,
                name="Fit",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=delay,
                y=lfilter(
                    fit.feedforward_taps[target],
                    fit.feedback_taps[target],
                    step_response,
                ),
                showlegend=True,
                name="IIR corrected",
            ),
            row=2,
            col=1,
        )

        fitting_report = table_html(
            table_dict(
                target,
                [
                    "Tau [us]",
                ],
                [fit.tau[target] / kilo],
            )
        )

    fig.update_layout(
        xaxis2_title="Delay [ns]",
        yaxis_title="Frequency [GHz]",
        yaxis2_title="Step response",
        showlegend=True,
        legend=dict(orientation="h"),
    )
    return [fig], fitting_report


def _update(
    results: LongCryoscopeResults, platform: CalibrationPlatform, qubit: QubitId
):
    """Update filters."""
    try:
        update.filters(
            amplitude=results.exp_amplitude[qubit],
            tau=results.tau[qubit],
            platform=platform,
            qubit=qubit,
        )
    except KeyError:
        log.info(f"Skipping filters update on qubit {qubit}.")


long_cryoscope = Routine(_acquisition, _fit, _plot, _update)
"""LongCryoscope Routine object."""
