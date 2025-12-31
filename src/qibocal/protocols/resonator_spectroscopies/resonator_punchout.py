from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, Parameter, PulseSequence, Sweeper

from qibocal import update
from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.result import magnitude

from ..utils import HZ_TO_GHZ, extract_feature, scaling_slice, table_dict, table_html

__all__ = ["resonator_punchout", "ResonatorPunchoutData"]


ResonatorPunchoutType = np.dtype(
    [("freq", np.float64), ("amp", np.float64), ("iq", np.float64, (2,))]
)


@dataclass
class ResonatorPunchoutParameters(Parameters):
    """ResonatorPunchout runcard inputs."""

    freq_width: int
    """Width for frequency sweep relative  to the readout frequency [Hz]."""
    freq_step: int
    """Frequency step for sweep [Hz]."""
    min_amp: float
    """Minimum amplitude."""
    max_amp: float
    """Maximum amplitude."""
    step_amp: float
    """Step amplitude."""


@dataclass
class ResonatorPunchoutResults(Results):
    """ResonatorPunchout outputs."""

    readout_frequency: dict[QubitId, float]
    """Readout frequency [GHz] for each qubit."""
    bare_frequency: Optional[dict[QubitId, float]]
    """Bare resonator frequency [GHz] for each qubit."""
    readout_amplitude: dict[QubitId, float]
    """Readout amplitude for each qubit."""
    successful_fit: dict[QubitId, bool]
    """flag for each qubit to see whether the fit was successful."""


@dataclass
class ResonatorPunchoutData(Data):
    """ResonatorPunchout data acquisition."""

    resonator_type: str
    """Resonator type."""
    amplitudes: dict[QubitId, list] = field(default_factory=dict)
    frequencies: dict[QubitId, list] = field(default_factory=dict)
    data: dict[QubitId, np.ndarray] = field(default_factory=dict)
    """Raw data acquired."""

    @property
    def find_min(self) -> bool:
        return self.resonator_type == "2D"

    def signal(self, qubit: QubitId) -> np.ndarray:
        return magnitude(self.data[qubit].iq)

    def grid(self, qubit: QubitId) -> tuple[np.ndarray]:
        x, y = np.meshgrid(self.frequencies[qubit], self.amplitudes)
        return x.ravel(), y.ravel(), self.signal(qubit).ravel()

    def normalized_signal(self, qubit: QubitId) -> np.ndarray:
        signal = self.signal(qubit).reshape(
            (
                len(np.unique(self.data[qubit].amp)),
                len(np.unique(self.data[qubit].freq)),
            )
        )
        return scaling_slice(signal, axis=1)

    def filtered_data(self, qubit: QubitId) -> tuple[np.ndarray]:
        x, y, _ = self.grid(qubit)
        return extract_feature(x, y, self.signal(qubit).ravel(), self.find_min)

    def register_qubit(self, qubit, iq, freq, amp):
        """Store output for single qubit."""
        size = len(freq) * len(amp)
        ar = np.empty(size, dtype=ResonatorPunchoutType)
        frequency, amplitudes = np.meshgrid(freq, amp)
        ar["freq"] = frequency.ravel()
        ar["amp"] = amplitudes.ravel()
        ar["iq"] = iq.reshape((size, 2))
        self.data[qubit] = np.rec.array(ar)


def _acquisition(
    params: ResonatorPunchoutParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> ResonatorPunchoutData:
    """Data acquisition for Punchout over amplitude."""
    # create a sequence of pulses for the experiment:
    # MZ

    # define the parameters to sweep and their range:
    # resonator frequency
    delta_frequency_range = np.arange(
        -params.freq_width / 2, params.freq_width / 2, params.freq_step
    )

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    ro_pulses = {}
    freq_sweepers = {}
    sequence = PulseSequence()
    for qubit in targets:
        natives = platform.natives.single_qubit[qubit]
        ro_channel, ro_pulse = natives.MZ()[0]
        ro_pulses[qubit] = ro_pulse.model_copy(
            update={"probe": ro_pulse.probe.model_copy(update={"amplitude": 1.0})}
        )
        sequence.append((ro_channel, ro_pulses[qubit]))

        probe = platform.qubits[qubit].probe
        f0 = platform.config(probe).frequency
        freq_sweepers[qubit] = Sweeper(
            parameter=Parameter.frequency,
            values=f0 + delta_frequency_range,
            channels=[probe],
        )

    amp_sweeper = Sweeper(
        parameter=Parameter.amplitude,
        range=(params.min_amp, params.max_amp, params.step_amp),
        pulses=[ro_pulses[qubit] for qubit in targets],
    )

    data = ResonatorPunchoutData(
        amplitudes=amp_sweeper.values.tolist(),
        frequencies={qubit: freq_sweepers[qubit].values.tolist() for qubit in targets},
        resonator_type=platform.resonator_type,
    )

    results = platform.execute(
        [sequence],
        [[amp_sweeper], [freq_sweepers[q] for q in targets]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    # retrieve the results for every qubit
    for qubit in targets:
        result = results[ro_pulses[qubit].id]
        data.register_qubit(
            qubit,
            iq=result,
            freq=freq_sweepers[qubit].values,
            amp=data.amplitudes,
        )

    return data


def _fit(data: ResonatorPunchoutData, fit_type="amp") -> ResonatorPunchoutResults:
    """Fit frequency and attenuation at high and low power for a given resonator."""

    low_freqs = {}
    high_freqs = {}
    ro_values = {}
    successful_fit = {}

    for qubit in data.qubits:
        filtered_x, filtered_y = data.filtered_data(qubit)

        if (
            filtered_x is None or filtered_y is None
        ):  # filtered_x and filtered_y have always the same shape
            successful_fit[qubit] = False
        else:
            # TODO: understand what is going on here
            if fit_type == "amp":
                best_freq = np.max(filtered_x)
                bare_freq = np.min(filtered_x)
            else:
                best_freq = np.min(filtered_x)
                bare_freq = np.max(filtered_x)
            ro_val = np.max(filtered_y[filtered_x == best_freq])

            low_freqs[qubit] = best_freq
            high_freqs[qubit] = bare_freq
            ro_values[qubit] = ro_val
            successful_fit[qubit] = True

    return ResonatorPunchoutResults(
        readout_frequency=low_freqs,
        bare_frequency=high_freqs,
        readout_amplitude=ro_values,
        successful_fit=successful_fit,
    )


def _plot(
    data: ResonatorPunchoutData, target: QubitId, fit: ResonatorPunchoutResults = None
):
    """Plotting function for ResonatorPunchout."""
    figures = []
    fitting_report = ""
    fig = go.Figure()
    x, y, _ = data.grid(target)
    fig.add_trace(
        go.Heatmap(
            x=x * HZ_TO_GHZ,
            y=y,
            z=data.normalized_signal(target).ravel(),
            colorbar=dict(title="Normalized signal"),
            colorscale="Viridis",
        )
    )
    filtered_x, filtered_y = data.filtered_data(target)

    if filtered_x is not None and filtered_y is not None:
        fig.add_trace(
            go.Scatter(
                x=filtered_x * HZ_TO_GHZ,
                y=filtered_y,
                mode="markers",
                name="Estimated points",
                marker=dict(color="rgb(248, 248, 248)"),
                showlegend=True,
            )
        )

    if fit is not None and fit.successful_fit[target]:
        # if fit.readout_frequency[target] is None then all the other two fields are None, one field
        # cannot be None with the remaining being not None
        fig.add_trace(
            go.Scatter(
                x=[
                    fit.readout_frequency[target] * HZ_TO_GHZ,
                ],
                y=[
                    fit.readout_amplitude[target],
                ],
                mode="markers",
                marker=dict(
                    size=8,
                    color="red",
                    symbol="circle",
                ),
                name="Estimated readout point",
                showlegend=True,
            )
        )
        fitting_report = table_html(
            table_dict(
                target,
                [
                    "Low Power Resonator Frequency [Hz]",
                    "Low Power readout amplitude [a.u.]",
                    "High Power Resonator Frequency [Hz]",
                ],
                [
                    np.round(fit.readout_frequency[target]),
                    np.round(fit.readout_amplitude[target], 3),
                    np.round(fit.bare_frequency[target]),
                ],
            )
        )

    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h"),
    )

    fig.update_xaxes(title_text="Frequency [GHz]")
    fig.update_yaxes(title_text="Amplitude [a.u.]")

    figures.append(fig)

    return figures, fitting_report


def _update(
    results: ResonatorPunchoutResults, platform: CalibrationPlatform, target: QubitId
):
    if results.successful_fit[target]:
        update.readout_frequency(results.readout_frequency[target], platform, target)
        update.dressed_resonator_frequency(
            results.readout_frequency[target], platform, target
        )
        update.bare_resonator_frequency(
            results.bare_frequency[target], platform, target
        )
        update.readout_amplitude(results.readout_amplitude[target], platform, target)


resonator_punchout = Routine(_acquisition, _fit, _plot, _update)
"""ResonatorPunchout Routine object."""
