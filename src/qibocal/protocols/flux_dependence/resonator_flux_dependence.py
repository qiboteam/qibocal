from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, Parameter, PulseSequence, Sweeper
from scipy.optimize import curve_fit

from qibocal.calibration import CalibrationPlatform

from ... import update
from ...auto.operation import Data, Protocol, QubitId, Results
from ...config import log
from ...result import magnitude
from ..utils import (
    GHZ_TO_HZ,
    HZ_TO_GHZ,
    readout_frequency,
    table_dict,
    table_html,
)
from . import utils

__all__ = ["ResonatorFluxParameters", "resonator_flux"]

from scipy.ndimage import median_filter
from scipy.signal import find_peaks
from scipy.special import erfinv

GAUSSIAN_FILTER1D_SIGMA = 2
INLIER_THRESHOLD = 0.5e6  # approximate width of a peak in the qubit spectroscopy in Hz
RANSAC_P_SUCCESS = (
    0.999  # desired probability of finding a sample containing only inliers
)
RANSAC_MIN_ITERATIONS = 100
RANSAC_MAX_ITERATIONS = 5000


@dataclass
class ResonatorFluxParameters(utils.FluxFrequencySweepParameters):
    """ResonatorFlux runcard inputs."""


@dataclass
class ResonatorFluxResults(Results):
    """ResonatoFlux outputs."""

    frequency: dict[QubitId, float] = field(default_factory=dict)
    """Readout frequency."""
    coupling: dict[QubitId, float] = field(default_factory=dict)
    """Qubit-resonator coupling."""
    asymmetry: dict[QubitId, float] = field(default_factory=dict)
    """Asymmetry between junctions."""
    sweetspot: dict[QubitId, float] = field(default_factory=dict)
    """Sweetspot for each qubit."""
    matrix_element: dict[QubitId, float] = field(default_factory=dict)
    """Sweetspot for each qubit."""
    fitted_parameters: dict[QubitId, float] = field(default_factory=dict)
    """Optimal parameters found from the fit,"""
    successful_fit: dict[QubitId, bool] = field(default_factory=dict)
    """flag for each qubit to see whether the fit was successful."""


ResFluxType = np.dtype(
    [
        ("freq", np.float64),
        ("bias", np.float64),
        ("signal", np.float64),
    ]
)
"""Custom dtype for resonator flux dependence."""


@dataclass
class ResonatorFluxData(Data):
    """ResonatorFlux acquisition outputs."""

    resonator_type: str
    """Resonator type."""
    qubit_frequency: dict[QubitId, float] = field(default_factory=dict)
    """Qubit frequencies."""
    bare_resonator_frequency: dict[QubitId, int] = field(default_factory=dict)
    """Qubit bare resonator frequency power provided by the user."""
    charging_energy: dict[QubitId, float] = field(default_factory=dict)
    """Qubit charging energy in Hz."""
    data: dict[QubitId, npt.NDArray[ResFluxType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, freq, bias, signal):
        """Store output for single qubit."""
        self.data[qubit] = utils.create_data_array(
            freq, bias, signal, dtype=ResFluxType
        )

    @property
    def find_min(self) -> bool:
        """Returns True if resonator_type is 2D else False otherwise."""
        return self.resonator_type == "2D"

    def filtered_data(self, qubit: QubitId) -> np.ndarray:
        """Apply mask to specific qubit data."""
        return utils.flux_extract_feature(
            self.data[qubit].freq,
            self.data[qubit].bias,
            self.data[qubit].signal,
            self.find_min,
        )


def _acquisition(
    params: ResonatorFluxParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> ResonatorFluxData:
    """Data acquisition for ResonatorFlux experiment."""

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    qubit_frequency = {}
    bare_resonator_frequency = {}
    charging_energy = {}
    matrix_element = {}
    offset = {}
    freq_sweepers = []
    offset_sweepers = []
    for q in targets:
        ro_sequence = platform.natives.single_qubit[q].MZ()
        ro_pulses[q] = ro_sequence[0][1]
        sequence += ro_sequence

        qubit = platform.qubits[q]
        offset0 = platform.config(qubit.flux).offset

        freq_sweepers.append(
            Sweeper(
                parameter=Parameter.frequency,
                values=readout_frequency(q, platform) + params.frequency_range,
                channels=[qubit.probe],
            )
        )
        offset_sweepers.append(
            Sweeper(
                parameter=Parameter.offset,
                values=offset0 + params.bias_range,
                channels=[qubit.flux],
            )
        )

        qubit_frequency[q] = platform.config(qubit.drive).frequency
        bare_resonator_frequency[q] = platform.calibration.single_qubits[
            q
        ].resonator.bare_frequency
        matrix_element[q] = platform.calibration.get_crosstalk_element(q, q)
        offset[q] = -offset0 * matrix_element[q]
        charging_energy[q] = platform.calibration.single_qubits[q].qubit.charging_energy

    data = ResonatorFluxData(
        resonator_type=platform.resonator_type,
        qubit_frequency=qubit_frequency,
        bare_resonator_frequency=bare_resonator_frequency,
        charging_energy=charging_energy,
    )
    results = platform.execute(
        [sequence],
        [offset_sweepers, freq_sweepers],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    # retrieve the results for every qubit
    for i, qubit in enumerate(targets):
        result = results[ro_pulses[qubit].id]
        data.register_qubit(
            qubit,
            signal=magnitude(result),
            freq=freq_sweepers[i].values,
            bias=offset_sweepers[i].values,
        )
    return data


@dataclass
class PeakCoordinates:
    bias: np.ndarray
    frequency: np.ndarray


def _extract_peak_coordinates(
    freq: np.ndarray,
    bias: np.ndarray,
    signal: np.ndarray,
) -> PeakCoordinates:
    """Extract the most prominent peaks per bias (if one is dominant enough)."""

    bias_pts, freq_pts = [], []
    signal_residuals = []
    is_peak = []
    for bias_val, row in zip(bias, signal):
        # There may be fluctuations along the frequency axis caused by elements such cables
        # or amplifiers. In principle this is flux independent and therefore ideal to remove
        # by subtracting the median per frequency bin. However, the arc may be very flat, in
        # which case we end up subtracting the arc rather than background. To avoid this, we
        # use median_filter.
        samples_per_peak = np.ceil(INLIER_THRESHOLD / np.diff(freq)[0])
        baseline = median_filter(row, size=int(20 * samples_per_peak), mode="nearest")
        residual = row - baseline

        # Estimate the std from median absolute deviation because a naive std is inflated by
        # the arc we're trying to detect.
        row_mad = np.median(np.abs(residual - np.median(residual)))
        row_std = 1.0 / (np.sqrt(2) * erfinv(0.5)) * row_mad

        # Detect both peaks and dips by finding prominent extrema in the absolute residual.
        peaks, props = find_peaks(np.abs(residual), prominence=row_std)
        if len(peaks) == 0:
            continue

        # Keep the most prominent extremum and record whether it is a peak or a dip.
        best = peaks[np.argmax(props["prominences"])]
        bias_pts.append(bias_val)
        freq_pts.append(freq[best])
        signal_residuals.append(residual)
        is_peak.append(residual[best] > 0)

    # Keep only the dominant extremum type to reject rows detecting the opposite feature.
    select_peaks = sum(is_peak) >= (len(is_peak) / 2)
    mask = np.equal(is_peak, select_peaks)

    bias_pts = np.asarray(bias_pts)[mask]
    freq_pts = np.asarray(freq_pts)[mask]

    return PeakCoordinates(
        bias=bias_pts,
        frequency=freq_pts,
    )


def _ransac_fit(
    freq_ghz: np.ndarray,
    bias_pts: np.ndarray,
    fit_function,
):
    """perform fit using RANSAC"""

    # The number of iterations is determined following the standard for RANSAC
    # https://en.wikipedia.org/wiki/Random_sample_consensus#Parameters
    N_needed = np.inf
    ransac_iterations = 0
    best_inliers = np.array([])
    best_params = np.array([])
    tried_subsets = set()
    while (
        ransac_iterations < min(N_needed, RANSAC_MAX_ITERATIONS)
        or ransac_iterations < RANSAC_MIN_ITERATIONS
    ):
        ransac_iterations += 1

        # randomly sample 6 points, because that's the dof of the parametrization
        subset = np.random.choice(len(bias_pts), 6, replace=False)
        subset_ = tuple(sorted(subset))
        if subset_ in tried_subsets:
            continue
        tried_subsets.add(subset_)

        try:
            popt, _ = curve_fit(
                fit_function,
                bias_pts[subset],
                freq_ghz[subset],
                method="lm",  # lm is a fast option
            )
        except RuntimeError:
            continue

        residuals_all = np.abs(freq_ghz - fit_function(bias_pts, *popt))
        inliers = residuals_all < INLIER_THRESHOLD * HZ_TO_GHZ
        if inliers.sum() == len(bias_pts):
            # all points are inliers, so we can proceed
            best_inliers = inliers
            best_params = popt
            break

        if inliers.sum() >= len(subset) and inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_params = popt
            denom = np.log(1 - (best_inliers.sum() / len(bias_pts)) ** len(subset))
            N_needed = np.log(1 - RANSAC_P_SUCCESS) / denom

    # Finally optimize by doing a least-squares fit to the best set of inliers
    popt, _ = curve_fit(
        fit_function,
        bias_pts[best_inliers],
        freq_ghz[best_inliers],
        p0=best_params,
        method="lm",
        maxfev=100000,
    )

    return popt


def _fit(data: ResonatorFluxData) -> ResonatorFluxResults:
    """PostProcessing for resonator_flux protocol.

    After applying a mask on the 2D data, the signal is fitted using
    the expected frequency vs flux behavior.
    The fitting procedure requires the knowledge of the bare resonator frequency,
    the charging energy Ec and the maximum qubit frequency which is assumed to be
    the frequency at which the qubit is placed.
    The protocol aims at extracting the sweetspot, the flux coefficient, the coupling,
    the asymmetry and the dressed resonator frequency.
    """

    coupling = {}
    resonator_freq = {}
    asymmetry = {}
    fitted_parameters = {}
    sweetspot = {}
    matrix_element = {}
    successful_fit = {}

    for qubit in data.qubits:
        qubit_data = data[qubit]

        freq, freq_idx = np.unique(qubit_data.freq, return_inverse=True)
        bias, bias_idx = np.unique(qubit_data.bias, return_inverse=True)
        signal = np.full((len(bias), len(freq)), np.nan)
        signal[bias_idx, freq_idx] = qubit_data.signal

        peak_coordinates = _extract_peak_coordinates(
            freq=freq,
            bias=bias,
            signal=signal,
        )

        def _fit_function(
            x: float,
            g: float,
            d: float,
            offset: float,
            normalization: float,
            freq: float,
            charging_energy: float,
        ):
            """Fit function for resonator flux dependence."""
            return utils.transmon_readout_frequency(
                xi=x,
                w_max=data.qubit_frequency[qubit] * HZ_TO_GHZ,
                xj=0,
                d=d,
                normalization=normalization,
                offset=offset,
                crosstalk_element=1,
                charging_energy=charging_energy,
                resonator_freq=freq,
                g=g,
            )

        try:
            popt = _ransac_fit(
                peak_coordinates.frequency * HZ_TO_GHZ,
                peak_coordinates.bias,
                fit_function=_fit_function,
            )
            fitted_parameters[qubit] = {
                "w_max": data.qubit_frequency[qubit] * HZ_TO_GHZ,
                "xj": 0,
                "d": popt[1],
                "normalization": popt[3],
                "offset": popt[2],
                "crosstalk_element": 1,
                "charging_energy": popt[5],
                "resonator_freq": popt[4],
                "g": popt[0],
            }
            matrix_element[qubit] = popt[3]
            sweetspot[qubit] = (np.round(popt[2]) - popt[2]) / popt[3]
            resonator_freq[qubit] = _fit_function(sweetspot[qubit], *popt) * GHZ_TO_HZ
            coupling[qubit] = popt[0]
            asymmetry[qubit] = popt[1]
            successful_fit[qubit] = True
        except ValueError as e:
            successful_fit[qubit] = False
            log.error(f"Error in resonator_flux protocol fit: {e} ")

    return ResonatorFluxResults(
        frequency=resonator_freq,
        coupling=coupling,
        matrix_element=matrix_element,
        sweetspot=sweetspot,
        asymmetry=asymmetry,
        fitted_parameters=fitted_parameters,
        successful_fit=successful_fit,
    )


def _plot(data: ResonatorFluxData, fit: ResonatorFluxResults, target: QubitId):
    """Plotting function for ResonatorFlux Experiment."""
    figures = utils.flux_dependence_plot(
        data, fit, target, utils.transmon_readout_frequency
    )

    if fit is not None and fit.successful_fit[target]:
        fitting_report = table_html(
            table_dict(
                target,
                [
                    "Coupling g [MHz]",
                    "Dressed resonator freq [Hz]",
                    "Asymmetry",
                    "Sweetspot [V]",
                    "Flux dependence [V]^-1",
                    "Chi [MHz]",
                ],
                [
                    np.round(fit.coupling[target] * 1e3, 2),
                    np.round(fit.frequency[target], 6),
                    np.round(fit.asymmetry[target], 3),
                    np.round(fit.sweetspot[target], 4),
                    np.round(fit.matrix_element[target], 4),
                    np.round(
                        (data.bare_resonator_frequency[target] - fit.frequency[target])
                        * 1e-6,
                        2,
                    ),
                ],
            )
        )
        return figures, fitting_report
    return figures, ""


def _update(
    results: ResonatorFluxResults, platform: CalibrationPlatform, qubit: QubitId
):
    if results.successful_fit[qubit]:
        update.dressed_resonator_frequency(results.frequency[qubit], platform, qubit)
        update.readout_frequency(results.frequency[qubit], platform, qubit)
        update.readout_coupling(results.coupling[qubit], platform, qubit)
        update.flux_offset(results.sweetspot[qubit], platform, qubit)
        update.sweetspot(results.sweetspot[qubit], platform, qubit)


resonator_flux = Protocol(_acquisition, _fit, _plot, _update)
"""ResonatorFlux Protocol object."""
