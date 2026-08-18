from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    PulseSequence,
    Sweeper,
)
from scipy.signal import find_peaks

from qibocal.auto.operation import Data, Protocol, QubitId, Results
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.result import magnitude
from qibocal.update import replace

from ... import update
from ..utils import (
    readout_frequency,
    table_dict,
    table_html,
)
from . import utils

__all__ = [
    "QubitFluxData",
    "QubitFluxParameters",
    "QubitFluxResults",
    "QubitFluxType",
    "qubit_flux",
]


@dataclass
class QubitFluxParameters(utils.FluxFrequencySweepParameters):
    """QubitFlux runcard inputs."""

    drive_amplitude: float = 0.01
    """Amplitude of the drive pulse."""
    drive_duration: int = 2000
    """Duration of the drive pulse."""


@dataclass
class QubitFluxResults(Results):
    """QubitFlux outputs."""

    sweetspot: dict[QubitId, float] = field(default_factory=dict)
    """Sweetspot for each qubit."""
    frequency: dict[QubitId, float] = field(default_factory=dict)
    """Drive frequency for each qubit."""
    fitted_parameters: dict[QubitId, dict[str, float]] = field(default_factory=dict)
    """Raw fitting output."""
    matrix_element: dict[QubitId, float] = field(default_factory=dict)
    """V_ii coefficient."""
    successful_fit: dict[QubitId, bool] = field(default_factory=dict)
    """flag for each qubit to see whether the fit was successful."""
    # The attributes below are only for visualizing the inliers/outliers for debugging
    peak_biases: dict[QubitId, list[float]] = field(default_factory=dict)
    """Bias of extracted peaks (for visualization)."""
    peak_frequencies: dict[QubitId, list[int]] = field(default_factory=dict)
    """Frequency of extracted peaks (for visualization)."""
    inliers: dict[QubitId, list[bool]] = field(default_factory=dict)
    """Boolean mask indicating which peaks are inliers (for visualization)."""


QubitFluxType = np.dtype(
    [
        ("freq", np.float64),
        ("bias", np.float64),
        ("signal", np.float64),
    ]
)
"""Custom dtype for resonator flux dependence."""


@dataclass
class QubitFluxData(Data):
    """QubitFlux acquisition outputs."""

    resonator_type: str
    """Resonator type."""
    charging_energy: dict[QubitId, float] = field(default_factory=dict)
    """Qubit charging energy."""
    qubit_frequency: dict[QubitId, float] = field(default_factory=dict)
    """Qubit charging energy."""
    data: dict[QubitId, npt.NDArray[QubitFluxType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, freq, bias, signal):
        """Store output for single qubit."""
        self.data[qubit] = utils.create_data_array(
            freq, bias, signal, dtype=QubitFluxType
        )


def _acquisition(
    params: QubitFluxParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> QubitFluxData:
    """Data acquisition for QubitFlux Experiment."""

    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    qubit_frequency = {}
    freq_sweepers = []
    offset_sweepers = []
    for q in targets:
        natives = platform.natives.single_qubit[q]
        qd_channel, qd_pulse = natives.RX()[0]
        ro_channel, ro_pulse = natives.MZ()[0]

        qd_pulse = replace(qd_pulse, duration=params.drive_duration)
        qd_pulse = replace(qd_pulse, amplitude=params.drive_amplitude)

        qd_pulses[q] = qd_pulse
        ro_pulses[q] = ro_pulse
        qubit_frequency[q] = frequency0 = platform.config(qd_channel).frequency

        sequence.append((qd_channel, qd_pulse))
        sequence.append((ro_channel, Delay(duration=qd_pulse.duration)))
        sequence.append((ro_channel, ro_pulse))

        # define the parameters to sweep and their range:
        freq_sweepers.append(
            Sweeper(
                parameter=Parameter.frequency,
                values=frequency0 + params.frequency_range,
                channels=[qd_channel],
            )
        )

        flux_channel = platform.qubits[q].flux
        offset0 = platform.config(flux_channel).offset
        offset_sweepers.append(
            Sweeper(
                parameter=Parameter.offset,
                values=offset0 + params.bias_range,
                channels=[flux_channel],
            )
        )

    data = QubitFluxData(
        resonator_type=platform.resonator_type,
        charging_energy={
            qubit: platform.calibration.single_qubits[qubit].qubit.charging_energy
            for qubit in targets
        },
        qubit_frequency=qubit_frequency,
    )
    results = platform.execute(
        [sequence],
        [offset_sweepers, freq_sweepers],
        updates=[
            {platform.qubits[q].probe: {"frequency": readout_frequency(q, platform)}}
            for q in targets
        ],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    for i, qubit in enumerate(targets):
        result = results[ro_pulses[qubit].id]
        data.register_qubit(
            qubit,
            signal=magnitude(result),
            freq=freq_sweepers[i].values,
            bias=offset_sweepers[i].values,
        )
    return data


def _extract_peak_coordinates(
    frequencies: npt.NDArray[np.floating],
    biases: npt.NDArray[np.floating],
    signal: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Extract the most prominent peaks in the qubit (flux,frequency) landscape. At most
    one peak per flux bin.
    """

    filtered_signal = utils.filter_data(signal)

    mad = np.median(
        np.abs(filtered_signal - np.median(filtered_signal, axis=1, keepdims=True))
    )
    height_limit = 3.5 * mad

    peak_biases, peak_frequencies = [], []
    for bias, row in zip(biases, filtered_signal):
        # Use find_peaks instead of argmax because there may be nothing in a row. Try
        # both peak and dip per row, since this may differ per row due to moving of the
        # resonator frequency.
        median_subtracted = row - np.median(row)
        peaks, peak_props = find_peaks(
            median_subtracted, height=height_limit, prominence=0
        )
        dips, dip_props = find_peaks(
            -median_subtracted, height=height_limit, prominence=0
        )
        if len(peaks) == 0 and len(dips) == 0:
            continue
        # Keep only the feature with the largest prominence per bias.
        if len(dips) == 0 or (
            len(peaks) > 0
            and peak_props["prominences"].max() >= dip_props["prominences"].max()
        ):
            best = peaks[np.argmax(peak_props["prominences"])]
        else:
            best = dips[np.argmax(dip_props["prominences"])]

        # Store bias and frequency of the peak.
        peak_biases.append(bias)
        peak_frequencies.append(frequencies[best])

    return np.asarray(peak_biases), np.asarray(peak_frequencies)


def _fit_function(data: QubitFluxData, qubit: QubitId):

    def func(x, w_max, normalization, offset):
        return utils.transmon_frequency(
            xi=x,
            w_max=w_max,
            xj=0,
            d=0,
            normalization=normalization,
            offset=offset,
            crosstalk_element=1,
            charging_energy=data.charging_energy[qubit],
        )

    return func


def _fit(data: QubitFluxData) -> QubitFluxResults:
    """
    Post-processing for QubitFlux Experiment. See `arXiv:0703002 <https://arxiv.org/abs/cond-mat/0703002>`_.
    Fit frequency as a function of current for the flux qubit spectroscopy data.
    All possible sweetspots :math:`x` are evaluated by the function
    :math:`x p_1 + p_2 = k`, for integers :math:`k`, where :math:`p_1` and :math:`p_2`
    are respectively the normalization and the offset, as defined in
    :mod:`qibocal.protocols.flux_dependence.utils.transmon_frequency`.
    The code returns the sweetspot that has the smallest absolute value within the data
    window, or else the nearest outside the window if it is within max_distance.
    """

    qubits = data.qubits
    frequency = {}
    sweetspot = {}
    matrix_element = {}
    fitted_parameters = {}
    successful_fit = {}
    peak_biases_dict = {}
    peak_frequencies_dict = {}
    inliers_dict = {}

    for qubit in qubits:
        qubit_data = data[qubit]

        freq = np.unique(qubit_data.freq)
        bias = np.unique(qubit_data.bias)
        signal = qubit_data.signal.reshape(len(bias), len(freq))

        peak_biases, peak_frequencies = _extract_peak_coordinates(
            frequencies=freq,
            biases=bias,
            signal=signal,
        )

        bounds = (
            [
                data.qubit_frequency[qubit] - 1e9,
                0,
                -1,
            ],
            [
                data.qubit_frequency[qubit] + 1e9,
                np.inf,
                1,
            ],
        )

        try:
            popt, inliers_mask = utils.ransac_fit(
                peak_biases,
                peak_frequencies,
                fit_function=_fit_function(data, qubit),
                # approximate width of a peak in the qubit spectroscopy
                residual_threshold=0.2e6,
                bounds=bounds,
            )

            fitted_parameters[qubit] = {
                "w_max": popt[0],
                "xj": 0,
                "d": 0,
                "normalization": popt[1],
                "offset": popt[2],
                "crosstalk_element": 1,
                "charging_energy": data.charging_energy[qubit],
            }
            frequency[qubit] = popt[0]
            sweetspot[qubit] = utils.select_sweetspot(
                popt[2],
                popt[1],
                (np.min(qubit_data.bias), np.max(qubit_data.bias)),
                max_distance=0.3,
            )
            matrix_element[qubit] = popt[1]
            successful_fit[qubit] = True

            # Store peak coordinates and inliers/outliers for plotting
            peak_biases_dict[qubit] = peak_biases.tolist()
            peak_frequencies_dict[qubit] = peak_frequencies.tolist()
            inliers_dict[qubit] = inliers_mask.tolist()

        except (ValueError, RuntimeError) as e:
            successful_fit[qubit] = False
            log.error(f"Error in qubit_flux protocol fit: {e}.")

    return QubitFluxResults(
        frequency=frequency,
        sweetspot=sweetspot,
        matrix_element=matrix_element,
        fitted_parameters=fitted_parameters,
        successful_fit=successful_fit,
        peak_biases=peak_biases_dict,
        peak_frequencies=peak_frequencies_dict,
        inliers=inliers_dict,
    )


def _plot(data: QubitFluxData, fit: QubitFluxResults, target: QubitId):
    """Plotting function for QubitFlux Experiment."""

    inliers_data = outliers_data = None
    if fit is not None and target in fit.peak_biases:
        peak_biases = np.asarray(fit.peak_biases.get(target, []))
        peak_frequencies = np.asarray(fit.peak_frequencies.get(target, []))
        coordinates_all_peaks = np.column_stack([peak_biases, peak_frequencies])

        inliers_mask = np.asarray(fit.inliers.get(target, []), dtype=bool)

        inliers_data = coordinates_all_peaks[inliers_mask]
        outliers_data = coordinates_all_peaks[~inliers_mask]

    figures = utils.flux_dependence_plot(
        data,
        fit,
        target,
        fit_function=utils.transmon_frequency,
        inliers=inliers_data,
        outliers=outliers_data,
    )

    if fit is not None and fit.successful_fit[target]:
        fitting_report = table_html(
            table_dict(
                target,
                [
                    "Sweetspot [a.u.]",
                    "Qubit Frequency at Sweetspot [Hz]",
                    "Flux dependence [a.u.]^-1",
                ],
                [
                    np.round(fit.sweetspot[target], 4),
                    np.round(fit.frequency[target], 4),
                    np.round(fit.matrix_element[target], 4),
                ],
            )
        )
        return figures, fitting_report
    return figures, ""


def _update(results: QubitFluxResults, platform: CalibrationPlatform, qubit: QubitId):
    if results.successful_fit[qubit]:
        update.drive_frequency(results.frequency[qubit], platform, qubit)
        platform.calibration.single_qubits[qubit].qubit.maximum_frequency = int(
            results.frequency[qubit]
        )
        update.sweetspot(results.sweetspot[qubit], platform, qubit)
        update.flux_offset(results.sweetspot[qubit], platform, qubit)
        update.crosstalk_matrix(results.matrix_element[qubit], platform, qubit, qubit)


qubit_flux = Protocol(_acquisition, _fit, _plot, _update)
"""QubitFlux Protocol object."""
