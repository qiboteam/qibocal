"""Rabi experiment that sweeps amplitude and frequency (with probability)."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Protocol, QubitId
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import (
    HZ_TO_GHZ,
    readout_frequency,
    table_dict,
    table_html,
)

from qibocal.result import collect, magnitude
from .amplitude_signal import RabiAmplitudeSignalResults, RabiAmplitudeSignalData
from .utils import fit_amplitude_function, rabi_initial_guess, sequence_amplitude, plot
from sklearn.decomposition import PCA


__all__ = [
    "RabiAmplitudeFreqSignalData",
    "RabiAmplitudeFrequencySignalParameters",
    "RabiAmplitudeSignalResults",
    "_update",
    "rabi_amplitude_frequency_signal",
]


@dataclass
class RabiAmplitudeFrequencySignalParameters(Parameters):
    """RabiAmplitudeFrequency runcard inputs."""

    min_amp: float
    """Minimum amplitude."""
    max_amp: float
    """Maximum amplitude."""
    step_amp: float
    """Step amplitude."""
    min_freq: int
    """Minimum frequency as an offset."""
    max_freq: int
    """Maximum frequency as an offset."""
    step_freq: int
    """Frequency to use as step for the scan."""
    rx90: bool = False
    """Calibration of native pi pulse, if true calibrates pi/2 pulse"""
    pulse_length: float | None = None
    """RX pulse duration [ns]."""


@dataclass
class RabiAmplitudeFrequencySignalResults(RabiAmplitudeSignalResults):
    """RabiAmplitudeFrequency outputs."""

    frequency: dict[QubitId, float] | dict[QubitId, list[float]]
    """Drive frequency for each qubit."""
    rx90: bool
    """Pi or Pi_half calibration"""


RabiAmpFreqSignalType = np.dtype(
    [
        ("amp", np.float64),
        ("freq", np.float64),
        ("i", np.float64),
        ("q", np.float64),
    ]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class RabiAmplitudeFreqSignalData(Data):
    """RabiAmplitudeFreqSignal data acquisition."""

    rx90: bool
    """Pi or Pi_half calibration"""
    durations: dict[QubitId, float] = field(default_factory=dict)
    """Pulse durations provided by the user."""
    data: dict[QubitId, npt.NDArray[RabiAmpFreqSignalType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""

    def register_qubit(self, qubit, freq, amp, i, q):
        """Store output for single qubit."""
        size = len(freq) * len(amp)
        frequency, amplitude = np.meshgrid(freq, amp)
        data = np.empty(size, dtype=RabiAmpFreqSignalType)
        data["freq"] = frequency.ravel()
        data["amp"] = amplitude.ravel()
        data["i"] = i.ravel()
        data["q"] = q.ravel()
        self.data[qubit] = np.rec.array(data)

    def amplitudes(self, qubit):
        """Unique qubit amplitudes."""
        return np.unique(self[qubit].amp)

    def frequencies(self, qubit):
        """Unique qubit frequency."""
        return np.unique(self[qubit].freq)

    def return_row_data(self, freq: float, qubit: QubitId):
        """Return the data subset for a selected drive frequency.

        Args:
            freq: Frequency value used to filter the recorded data.
            qubit: Identifier of the qubit whose data should be returned.

        Returns:
            The row data restricted to the requested frequency.
        """

        selected_freq_data = self.data[qubit][self.data[qubit].freq == freq]

        return RabiAmplitudeSignalData(
            rx90=self.rx90,
            durations=self.durations,
            data={qubit:selected_freq_data}
        )


def _acquisition(
    params: RabiAmplitudeFrequencySignalParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> RabiAmplitudeFreqSignalData:
    """Data acquisition for Rabi experiment sweeping amplitude."""

    sequence, qd_pulses, ro_pulses, durations = sequence_amplitude(
        targets, params, platform, params.rx90
    )

    frequency_range = np.arange(
        params.min_freq,
        params.max_freq,
        params.step_freq,
    )
    freq_sweepers = {}
    for qubit in targets:
        channel = platform.qubits[qubit].drive
        freq_sweepers[qubit] = Sweeper(
            parameter=Parameter.frequency,
            values=platform.config(channel).frequency + frequency_range,
            channels=[channel],
        )
    amp_sweeper = Sweeper(
        parameter=Parameter.amplitude,
        range=(params.min_amp, params.max_amp, params.step_amp),
        pulses=[qd_pulses[qubit] for qubit in targets],
    )

    data = RabiAmplitudeFreqSignalData(durations=durations, rx90=params.rx90)

    results = platform.execute(
        [sequence],
        [[amp_sweeper], [freq_sweepers[q] for q in targets]],
        updates=[
            {platform.qubits[q].probe: {"frequency": readout_frequency(q, platform)}}
            for q in targets
        ],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    for qubit in targets:
        result = results[ro_pulses[qubit].id]
        data.register_qubit(
            qubit=qubit,
            freq=freq_sweepers[qubit].values,
            amp=amp_sweeper.values,
            i=result[..., 0],
            q=result[..., 1],
        )
    return data


def _fit(data: RabiAmplitudeFreqSignalData) -> RabiAmplitudeFrequencySignalResults:
    """Do not perform any fitting procedure."""
    fitted_frequencies = {}
    fitted_amplitudes = {}
    fitted_parameters = {}

    for qubit in data.data:
        amps = data.amplitudes(qubit)
        freqs = data.frequencies(qubit)

        quadratures = collect(data[qubit].i, data[qubit].q)
        quadratures_matrix = quadratures.reshape(len(amps), len(freqs), -1)
        quadratures_matrix = np.moveaxis(quadratures_matrix, 0, 1) 

        # computing PCA for each frequency value and only take the most relevant component
        pc_matrix = np.asarray([PCA().fit_transform(x)[:, 0] for x in quadratures_matrix])
        # guess optimal frequency maximizing oscillation amplitude
        # here pc_matrix has dimensions (n_freqs, n_amps), so we need to compute 
        # initial guesses over axis==1
        full_pguesses = rabi_initial_guess(amps, pc_matrix, "amp", signal=True, axis=1)

        # guess has the following elements:
        # 0. median guess
        # 1. amplitude guess
        # 2. period guess
        # 3. phase guess
        # we estimate the best frequency by maximizing the amplitude estimation
        index = np.argmax(full_pguesses[1])
        breakpoint()

        frequency = freqs[index]
        y = pc_matrix[index]

        # initial guesses for the best frequency row
        pguess = [p[index] for p in full_pguesses]
        try:
            popt, _, pi_pulse_parameter = fit_amplitude_function(
                amps,
                y,
                pguess,
                signal=True,
            )
            fitted_frequencies[qubit] = frequency
            fitted_amplitudes[qubit] = pi_pulse_parameter
            fitted_parameters[qubit] = popt

        except Exception as e:
            log.warning(f"Rabi fit failed for qubit {qubit} due to {e}.")

    return RabiAmplitudeFrequencySignalResults(
        amplitude=fitted_amplitudes,
        length=data.durations,
        fitted_parameters=fitted_parameters,
        frequency=fitted_frequencies,
        rx90=data.rx90,
    )


def _plot(
    data: RabiAmplitudeFreqSignalData,
    target: QubitId,
    fit: RabiAmplitudeFrequencySignalResults | None = None,
):
    """Plotting function for RabiAmplitudeFrequency."""
    figures = []
    fitting_report = ""
    fig = go.Figure()
    qubit_data = data[target]
    frequencies = qubit_data.freq * HZ_TO_GHZ
    amplitudes = qubit_data.amp

    quadratures_matrix = collect(qubit_data.i, qubit_data.q).reshape(
        len(data.amplitudes(target)),
        len(data.frequencies(target)),
        -1
    )
    quadratures_matrix = np.moveaxis(quadratures_matrix, 0, 1)

    # computing PCA for each frequency value and only take the most relevant component
    pc_matrix = np.asarray([PCA().fit_transform(x)[:, 0] for x in quadratures_matrix]).T
    # pc_matrix = np.asarray([magnitude(x) for x in quadratures_matrix]).T
    
    fig.add_trace(
        go.Heatmap(
            x=amplitudes,
            y=frequencies,
            z=pc_matrix.ravel(),
            colorbar_x=1.0,
        ),
    )
    fig.update_layout(
            title="Rabi 2D IQ Signal",
            xaxis_title="Amplitude [a.u.]",
            yaxis_title="Frequency [GHz]",
    )

    if fit is not None:
        selected_frequency =  fit.frequency[target]

        fig.add_trace(
            go.Scatter(
                x=[min(amplitudes), max(amplitudes)],
                y=[selected_frequency * HZ_TO_GHZ] * 2,
                mode="lines",
                line={"color": "white", "width": 4, "dash": "dash"},
            ),
        )
        pulse_name = "Pi-half pulse" if data.rx90 else "Pi pulse"

        fitting_report = table_html(
            table_dict(
                target,
                ["Optimal rabi frequency", f"{pulse_name} amplitude"],
                [
                    fit.frequency[target],
                    f"{fit.amplitude[target]:.6f} [a.u]",
                ],
            )
        )

        fitted_data = data.return_row_data(selected_frequency, target) 
        rabi1d_figure, rabi1d_report = plot(fitted_data, target, fit, data.rx90)
        fitting_report += rabi1d_report
        figures.extend(rabi1d_figure)

    figures.insert(0, fig)

    return figures, fitting_report


def _update(
    results: RabiAmplitudeFrequencySignalResults,
    platform: CalibrationPlatform,
    target: QubitId,
):
    update.drive_duration(results.length[target], results.rx90, platform, target)
    update.drive_amplitude(results.amplitude[target], results.rx90, platform, target)
    update.drive_frequency(results.frequency[target], platform, target)


rabi_amplitude_frequency_signal = Protocol(_acquisition, _fit, _plot, _update)
"""Rabi amplitude with frequency tuning."""
