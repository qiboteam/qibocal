"""Rabi experiment that sweeps length and frequency."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper
from sklearn.decomposition import PCA

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Protocol, QubitId
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import table_dict, table_html

from ...result import collect
from ..utils import HZ_TO_GHZ, readout_frequency
from .length_signal import RabiLengthSignalResults
from .utils import fit_length_function, plot, rabi_initial_guess, sequence_length

__all__ = [
    "RabiLengthFreqSignalData",
    "RabiLengthFrequencySignalParameters",
    "RabiLengthFrequencySignalResults",
    "_update",
    "rabi_length_frequency_signal",
]


@dataclass
class RabiLengthFrequencySignalParameters(Parameters):
    """RabiLengthFrequency runcard inputs."""

    pulse_duration_start: float
    """Initial pi pulse duration [ns]."""
    pulse_duration_end: float
    """Final pi pulse duration [ns]."""
    pulse_duration_step: float
    """Step pi pulse duration [ns]."""
    min_freq: int
    """Minimum frequency as an offset."""
    max_freq: int
    """Maximum frequency as an offset."""
    step_freq: int
    """Frequency to use as step for the scan."""
    pulse_amplitude: float | None = None
    """Pi pulse amplitude. Same for all qubits."""
    rx90: bool = False
    """Calibration of native pi pulse, if true calibrates pi/2 pulse"""
    interpolated_sweeper: bool = False
    """Use real-time interpolation if supported by instruments."""


@dataclass
class RabiLengthFrequencySignalResults(RabiLengthSignalResults):
    """RabiLengthFrequency outputs."""

    rx90: bool
    """Pi or Pi_half calibration"""
    frequency: dict[QubitId, float]
    """Drive frequency for each qubit."""


RabiLenFreqSignalType = np.dtype(
    [
        ("len", np.float64),
        ("freq", np.float64),
        ("i", np.float64),
        ("q", np.float64),
    ]
)
"""Custom dtype for rabi length."""


@dataclass
class RabiLengthFreqSignalData(Data):
    """RabiLengthFreqSignal data acquisition."""

    rx90: bool
    """Pi or Pi_half calibration"""
    amplitudes: dict[QubitId, float] = field(default_factory=dict)
    """Pulse amplitudes provided by the user."""
    data: dict[QubitId, npt.NDArray[RabiLenFreqSignalType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""

    def register_qubit(self, qubit, freq, lens, signal, phase):
        """Store output for single qubit."""
        size = len(freq) * len(lens)
        frequency, length = np.meshgrid(freq, lens)
        data = np.empty(size, dtype=RabiLenFreqSignalType)
        data["freq"] = frequency.ravel()
        data["len"] = length.ravel()
        data["signal"] = signal.ravel()
        data["phase"] = phase.ravel()
        self.data[qubit] = np.rec.array(data)

    def durations(self, qubit) -> npt.NDArray:
        """Unique qubit lengths."""
        return np.unique(self[qubit].len)

    def frequencies(self, qubit) -> npt.NDArray:
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

        return RabiLengthFreqSignalData(
            rx90=self.rx90, amplitudes=self.amplitudes, data={qubit: selected_freq_data}
        )


def _acquisition(
    params: RabiLengthFrequencySignalParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> RabiLengthFreqSignalData:
    """Data acquisition for Rabi experiment sweeping length."""

    sequence, qd_pulses, delays, ro_pulses, amplitudes = sequence_length(
        targets, params, platform, params.rx90
    )

    sweep_range = (
        params.pulse_duration_start,
        params.pulse_duration_end,
        params.pulse_duration_step,
    )
    if params.interpolated_sweeper:
        len_sweeper = Sweeper(
            parameter=Parameter.duration_interpolated,
            range=sweep_range,
            pulses=[qd_pulses[q] for q in targets],
        )
    else:
        len_sweeper = Sweeper(
            parameter=Parameter.duration,
            range=sweep_range,
            pulses=[qd_pulses[q] for q in targets] + [delays[q] for q in targets],
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

    data = RabiLengthFreqSignalData(amplitudes=amplitudes, rx90=params.rx90)

    results = platform.execute(
        [sequence],
        [[len_sweeper], [freq_sweepers[q] for q in targets]],
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
            lens=len_sweeper.values,
            signal=result[..., 0],
            phase=result[..., 1],
        )
    return data


def _fit(data: RabiLengthFreqSignalData) -> RabiLengthFrequencySignalResults:
    """Do not perform any fitting procedure."""
    fitted_frequencies: dict[QubitId, float] = {}
    fitted_durations: dict[QubitId, float] = {}
    fitted_parameters: dict[QubitId, list[float]] = {}

    for qubit in data.data:
        durations = data.durations(qubit)
        freqs = data.frequencies(qubit)

        quadratures = collect(data[qubit].i, data[qubit].q)
        quadratures_matrix = quadratures.reshape(len(durations), len(freqs), -1)
        quadratures_matrix = np.moveaxis(quadratures_matrix, 0, 1)

        # computing PCA for each frequency value and only take the most relevant component
        pc_matrix = np.asarray(
            [PCA().fit_transform(x)[:, 0] for x in quadratures_matrix]
        )
        # guess optimal frequency maximizing oscillation amplitude
        # here pc_matrix has dimensions (n_freqs, n_amps), so we need to compute
        # initial guesses over axis==1
        full_pguesses = rabi_initial_guess(
            durations, pc_matrix, "length", signal=True, axis=1
        )

        # guess has the following elements:
        # 0. median guess
        # 1. amplitude guess
        # 2. period guess
        # 3. phase guess
        # 4. decaying constant guess
        # we estimate the best frequency by maximizing the amplitude estimation
        index = np.argmax(full_pguesses[1])

        frequency = freqs[index]
        y = pc_matrix[index]

        # initial guesses for the best frequency row
        pguess = [p[index] for p in full_pguesses]
        try:
            popt, _, pi_pulse_parameter = fit_length_function(durations, y, pguess)
            fitted_frequencies[qubit] = frequency
            fitted_durations[qubit] = pi_pulse_parameter
            fitted_parameters[qubit] = popt

        except Exception as e:
            log.warning(f"Rabi fit failed for qubit {qubit} due to {e}.")

    return RabiLengthFrequencySignalResults(
        length=fitted_durations,
        amplitude=data.amplitudes,
        fitted_parameters=fitted_parameters,
        frequency=fitted_frequencies,
        rx90=data.rx90,
    )


def _plot(
    data: RabiLengthFreqSignalData,
    target: QubitId,
    fit: RabiLengthFrequencySignalResults | None = None,
):
    """Plotting function for RabiLengthFrequency."""
    figures = []
    fitting_report = ""
    fig = go.Figure()
    qubit_data = data[target]
    frequencies = qubit_data.freq * HZ_TO_GHZ
    durations = qubit_data.len

    quadratures_matrix = collect(qubit_data.i, qubit_data.q).reshape(
        len(data.durations(target)), len(data.frequencies(target)), -1
    )
    quadratures_matrix = np.moveaxis(quadratures_matrix, 0, 1)

    # computing PCA for each frequency value and only take the most relevant component
    pc_matrix = np.asarray([PCA().fit_transform(x)[:, 0] for x in quadratures_matrix]).T

    fig.add_trace(
        go.Heatmap(
            x=durations,
            y=frequencies,
            z=pc_matrix.ravel(),
            colorbar_x=1.0,
        ),
    )
    fig.update_layout(
        title="Rabi 2D IQ Signal",
        xaxis_title="Time [ns]",
        yaxis_title="Frequency [GHz]",
    )

    if fit is not None:
        selected_frequency = fit.frequency[target]

        fig.add_trace(
            go.Scatter(
                x=[min(durations), max(durations)],
                y=[selected_frequency * HZ_TO_GHZ] * 2,
                mode="lines",
                line={"color": "white", "width": 4, "dash": "dash"},
            ),
        )
        pulse_name = "Pi-half pulse" if data.rx90 else "Pi pulse"

        fitting_report = table_html(
            table_dict(
                target,
                ["Optimal rabi frequency", f"{pulse_name} duration"],
                [
                    fit.frequency[target],
                    f"{fit.length[target]:.6f} [ns]",
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
    results: RabiLengthFrequencySignalResults,
    platform: CalibrationPlatform,
    target: QubitId,
):
    update.drive_amplitude(results.amplitude[target], results.rx90, platform, target)
    update.drive_duration(results.length[target], results.rx90, platform, target)
    update.drive_frequency(results.frequency[target], platform, target)


rabi_length_frequency_signal = Protocol(_acquisition, _fit, _plot, _update)
"""Rabi length with frequency tuning."""
