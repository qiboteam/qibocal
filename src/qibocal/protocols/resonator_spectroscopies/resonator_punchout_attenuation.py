from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, Parameter, PulseSequence, Sweeper

from qibocal import update
from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.result import magnitude, phase

from ..utils import HZ_TO_GHZ, reshaping_raw_signal, table_dict, table_html
from .resonator_utils import fit_punchout, punchout_extract_feature, punchout_mask

__all__ = ["resonator_punchout_attenuation", "ResonatorPunchoutAttenuationData"]


@dataclass
class ResonatorPunchoutAttenuationParameters(Parameters):
    """ResonatorPunchoutAttenuation runcard inputs."""

    freq_width: int
    """Width for frequency sweep relative to the readout frequency [Hz]."""
    freq_step: int
    """Frequency step for sweep [Hz]."""
    min_attenuation: float
    """Minimum LO attenuation [dB] (0 = no attenuation, lower = more attenuation)."""
    max_attenuation: float
    """Maximum LO attenuation [dB]."""
    step_attenuation: float
    """Step LO attenuation [dB]."""
    attenuation_range: np.ndarray = None

    def compute_attenuation_range(
        self, platform: CalibrationPlatform
    ) -> npt.NDArray[np.float64]:
        """LO attenuation range [dB]."""
        self.attenuation_range = np.arange(
            self.min_attenuation,
            self.max_attenuation,
            self.step_attenuation,
        )
        if "qm" in platform.instruments:
            self.attenuation_range *= -1

    @property
    def delta_frequency_range(self) -> npt.NDArray[np.float64]:
        """Frequency detuning range [Hz]."""
        return np.arange(
            -self.freq_width / 2,
            self.freq_width / 2 + self.freq_step,
            self.freq_step,
        )


@dataclass
class ResonatorPunchoutAttenuationResults(Results):
    """ResonatorPunchoutAttenuation outputs."""

    readout_frequency: dict[QubitId, float]
    """Readout frequency [GHz] for each qubit."""
    bare_frequency: Optional[dict[QubitId, float]]
    """Bare resonator frequency [GHz] for each qubit."""
    readout_attenuation: dict[QubitId, float]
    """Readout LO attenuation [dB] for each qubit."""
    successful_fit: dict[QubitId, bool]
    """flag for each qubit to see whether the fit was successful."""


@dataclass
class ResonatorPunchoutAttenuationData(Data):
    """ResonatorPunchoutAttenuation data acquisition."""

    resonator_type: str
    """Resonator type."""
    # TODO: maybe we want different attenuations for each RO line (if possible) ?
    attenuations: list = None
    """LO attenuations provided by the user."""
    frequencies: dict[QubitId, list] = field(default_factory=dict)
    data: dict[QubitId, np.ndarray] = field(default_factory=dict)
    """Raw data acquired, IQ components of the readout signal."""

    @property
    def find_min(self):
        return self.resonator_type == "2D"

    def signal(self, qubit: QubitId) -> np.ndarray:
        return magnitude(self.data[qubit])

    def phase(self, qubit: QubitId) -> np.ndarray:
        return phase(self.data[qubit])

    def grid(self, qubit: QubitId) -> tuple[np.ndarray]:
        x, y = np.meshgrid(self.frequencies[qubit], self.attenuations)
        return x.ravel(), y.ravel(), self.signal(qubit).ravel()

    def filtered_data(self, qubit: QubitId) -> tuple[np.array, np.array]:
        x, y, z = self.grid(qubit)
        return punchout_extract_feature(x, y, z, self.find_min)


def _acquisition(
    params: ResonatorPunchoutAttenuationParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> ResonatorPunchoutAttenuationData:
    """Data acquisition for Punchout using LO attenuation sweep.
    Arguments:
        params: ResonatorPunchoutAttenuationParameters
        platform: CalibrationPlatform
        targets: List of qubits to calibrate
    Returns:
        ResonatorPunchoutAttenuationData
    """

    assert params.min_attenuation >= 0, """minimum attenuation value is defined >=0"""
    assert params.max_attenuation >= params.min_attenuation, (
        """max_attenuation is always >= min_attenuation"""
    )
    assert params.step_attenuation >= 0, """step_attenuation is always >=0"""

    # compute range of attenuation to sweep on
    params.compute_attenuation_range(platform)

    # Get readout LO channels for each qubit
    ro_los = {}
    original_attenuations = {}

    sequence = PulseSequence()
    ro_pulses = {}
    freq_sweepers = {}
    for qubit in targets:
        # Create pulse sequence
        natives = platform.natives.single_qubit[qubit]
        ro_channel, ro_pulse = natives.MZ()[0]
        ro_pulses[qubit] = ro_pulse
        sequence.append((ro_channel, ro_pulse))

        # Get the LO channel for this readout
        probe = platform.qubits[qubit].probe
        lo_channel = platform.channels[probe].lo
        ro_los[qubit] = lo_channel

        # Create frequency sweeper
        f0 = platform.config(probe).frequency
        freq_sweepers[qubit] = Sweeper(
            parameter=Parameter.frequency,
            values=f0 + params.delta_frequency_range,
            channels=[probe],
        )

        # Store original attenuation (power in qibocal is actually attenuation)
        original_attenuations[qubit] = platform.config(lo_channel).power

    for qubit in targets:
        # Get the LO channel for this readout
        probe = platform.qubits[qubit].probe
        lo_channel = platform.channels[probe].lo
        ro_los[qubit] = lo_channel

        # Store original attenuation (power in qibocal is actually attenuation)
        original_attenuations[qubit] = platform.config(lo_channel).power

    data = ResonatorPunchoutAttenuationData(
        attenuations=(params.attenuation_range).tolist(),
        resonator_type=platform.resonator_type,
    )

    # Loop over attenuation values
    for attenuation in params.attenuation_range:
        # Update LO attenuation for all target qubits
        updates = []
        for qubit in targets:
            lo_channel = ro_los[qubit]
            updates.append({lo_channel: {"power": attenuation}})

        # Execute with frequency sweep only
        results = platform.execute(
            [sequence],
            [[freq_sweepers[q] for q in targets]],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
            updates=updates,
        )

        # Collect results for this attenuation level
        for qubit in targets:
            measure = np.expand_dims(results[ro_pulses[qubit].id], axis=0)
            if qubit not in data.data:
                data.data[qubit] = measure
            else:
                data.data[qubit] = np.concatenate((data.data[qubit], measure), axis=0)

    data.frequencies = {
        qubit: freq_sweepers[qubit].values.tolist() for qubit in targets
    }

    return data


def _fit(data: ResonatorPunchoutAttenuationData) -> ResonatorPunchoutAttenuationResults:
    """Fit frequency and attenuation at high and low power for a given resonator."""

    readout_freqs = {}
    bare_freqs = {}
    ro_values = {}
    successful_fit = {}

    for qubit in data.qubits:
        filtered_x, filtered_y = data.filtered_data(qubit)
        bare_freq, readout_freq, ro_val, fit_flag = fit_punchout(filtered_x, filtered_y)

        if fit_flag:
            readout_freqs[qubit] = readout_freq
            bare_freqs[qubit] = bare_freq
            ro_values[qubit] = ro_val
        successful_fit[qubit] = fit_flag

    return ResonatorPunchoutAttenuationResults(
        readout_frequency=readout_freqs,
        bare_frequency=bare_freqs,
        readout_attenuation=ro_values,
        successful_fit=successful_fit,
    )


def _plot(
    data: ResonatorPunchoutAttenuationData,
    fit: ResonatorPunchoutAttenuationResults,
    target: QubitId,
):
    """Plotting function for ResonatorPunchoutAttenuation."""
    figures = []
    fitting_report = ""

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
        subplot_titles=(
            "Normalised Signal [a.u.]",
            "Phase [rad]",
        ),
    )

    frequencies, attenuations, qubit_signal = data.grid(target)
    _, _, qubit_signal = reshaping_raw_signal(frequencies, attenuations, qubit_signal)
    qubit_signal = punchout_mask(qubit_signal).ravel()
    qubit_phase = data.phase(target).ravel()
    frequencies *= HZ_TO_GHZ

    fig.add_trace(
        go.Heatmap(
            x=frequencies,
            y=attenuations,
            z=qubit_signal,
            colorbar_x=0.46,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            x=frequencies,
            y=attenuations,
            z=qubit_phase,
            colorbar_x=1.01,
        ),
        row=1,
        col=2,
    )

    filtered_x, filtered_y = data.filtered_data(target)
    if filtered_x is not None and filtered_y is not None:
        fig.add_trace(
            go.Scatter(
                x=filtered_x * HZ_TO_GHZ,
                y=filtered_y,
                mode="markers",
                name="Estimated points",
                marker=dict(color="rgb(248, 0, 0)"),
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # Flip y-axis so low attenuation (high power) is at top
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)

    if fit is not None and fit.successful_fit[target]:
        fig.add_trace(
            go.Scatter(
                x=[fit.readout_frequency[target] * HZ_TO_GHZ],
                y=[fit.readout_attenuation[target]],
                mode="markers",
                marker=dict(
                    size=8,
                    color="green",
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
                    "Low Power Readout LO Attenuation [dB]",
                    "High Power Resonator Frequency [Hz]",
                ],
                [
                    np.round(fit.readout_frequency[target]),
                    np.round(fit.readout_attenuation[target], 2),
                    np.round(fit.bare_frequency[target]),
                ],
            )
        )

    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h"),
    )

    fig.update_xaxes(title_text="Frequency [GHz]", row=1, col=1)
    fig.update_xaxes(title_text="Frequency [GHz]", row=1, col=2)
    fig.update_yaxes(title_text="LO Attenuation [dB]", row=1, col=1)
    figures.append(fig)

    return figures, fitting_report


def _update(
    results: ResonatorPunchoutAttenuationResults,
    platform: CalibrationPlatform,
    target: QubitId,
):
    """Update platform with fitted parameters if fit was successful."""
    if results.successful_fit[target]:
        update.readout_frequency(results.readout_frequency[target], platform, target)
        update.bare_resonator_frequency(
            results.bare_frequency[target], platform, target
        )
        update.dressed_resonator_frequency(
            results.readout_frequency[target], platform, target
        )
        update.lo_attenuation(
            results.readout_attenuation[target],
            platform,
            target,
            channel_type="probe",
        )


resonator_punchout_attenuation = Routine(_acquisition, _fit, _plot, _update)
"""**Resonator Punchout Attenuation Qibocal Routine Object.**

This routine performs a resonator punchout (power shift) measurement by sweeping the LO attenuation
and the IF frequency to determine the critical power for a qubit's resonator to be dispersively shifted.

At low power, the effective resonator frequency is shifted by χ due to the qubit state, while at high power, the resonator frequency
approaches its bare frequency.

.. math::
    \\omega_{dressed} = \\omega_{bare} + \\chi

In general the frequency of the peak in the resonator can be approximated by

.. math::
    \\omega_{\text{peak}}(\bar{n}) = \\omega_r + \\chi \frac{1}{1 + \bar{n}/n_{\text{crit}}}

Where :math:`n_{\\text{crit}} = \\Delta^2 / 4g^2` is the critical photon number at which the resonator frequency starts to shift towards the bare frequency.
"""
