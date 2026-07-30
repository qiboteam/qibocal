from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, Parameter, PulseSequence, Sweeper
from scipy.signal import find_peaks

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


# approximate width of a peak in the resonator spectroscopy in Hz
APPROXIMATE_RESONATOR_PEAK_WIDTH = 0.2e6


@dataclass
class ResonatorFluxParameters(utils.FluxFrequencySweepParameters):
    """ResonatorFlux runcard inputs."""

    bias_center: float | None = None
    freq_center: float | None = None


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


def _extract_peak_coordinates(
    freq: npt.NDArray[np.float64],
    bias: npt.NDArray[np.float64],
    signal: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Extract the most prominent peaks in the resonator (flux,frequency) landscape. At
    most one peak per flux bin.
    """
    # We remove the median signal per frequency since in the case of the resonator
    # there is a frequency-dependent but flux-independent background signal.
    # sometimes there are bright spots for a given bias. Not sure what causes them,
    # but this hopefully gets rid of them.
    median_per_flux = np.median(signal, axis=1, keepdims=True)
    median_per_frequency = np.median(signal, axis=0, keepdims=True)
    global_median = np.median(signal)

    centered_signal = signal - median_per_flux - median_per_frequency + global_median
    bias_pts, freq_pts = [], []
    is_peak = []
    for bias_val, row in zip(bias, centered_signal):
        # Detect both peaks and dips by finding prominent extrema in the absolute
        # residual
        peaks, props = find_peaks(np.abs(row), prominence=0)
        if len(peaks) == 0:
            continue

        # Keep the most prominent extremum, along with its prominence, and whether it is
        # a peak or dip
        best = peaks[np.argmax(props["prominences"])]
        bias_pts.append(bias_val)
        freq_pts.append(freq[best])
        is_peak.append(row[best] > 0)

    # Keep only the dominant extremum type and ignore extrema of the opposite feature
    select_peaks = sum(is_peak) >= (len(is_peak) / 2)
    mask = np.equal(is_peak, select_peaks)
    bias_pts = np.asarray(bias_pts)[mask]
    freq_pts = np.asarray(freq_pts)[mask]

    return bias_pts, freq_pts


def _fit_function(data: ResonatorFluxData, qubit: QubitId):

    def func(
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

    return func


def _fit(data: ResonatorFluxData) -> ResonatorFluxResults:
    """PostProcessing for resonator_flux protocol.

    The fitting procedure requires the knowledge of the bare resonator frequency, the
    charging energy Ec and the maximum qubit frequency which is assumed to be the
    frequency at which the qubit is placed.

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

        peak_biases, peak_frequencies = _extract_peak_coordinates(
            freq=freq,
            bias=bias,
            signal=signal,
        )

        w_max = data.qubit_frequency[qubit] * HZ_TO_GHZ
        fit_function = _fit_function(data, qubit)

        # bounds for (g, d, offset, normalization, freq, charging_energy)
        bare_resonator_freq = data.bare_resonator_frequency[qubit] * HZ_TO_GHZ
        bounds = (
            [0, 0, -1, 0, bare_resonator_freq - 0.5, 0],
            [
                0.5,
                1,
                1,
                np.inf,
                bare_resonator_freq + 0.5,
                data.charging_energy[qubit] * HZ_TO_GHZ + 0.3,
            ],
        )
        try:
            popt = utils.ransac_fit(
                peak_biases,
                peak_frequencies * HZ_TO_GHZ,
                fit_function=fit_function,
                residual_threshold=APPROXIMATE_RESONATOR_PEAK_WIDTH * HZ_TO_GHZ,
                bounds=bounds,
            )
            fitted_parameters[qubit] = {
                "w_max": w_max,
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
            sweetspot[qubit] = utils.select_sweetspot(
                popt[2],
                popt[3],
                (np.min(data[qubit].bias), np.max(data[qubit].bias)),
                max_distance=0.3,
            )
            resonator_freq[qubit] = fit_function(sweetspot[qubit], *popt) * GHZ_TO_HZ
            coupling[qubit] = popt[0]
            asymmetry[qubit] = popt[1]
            successful_fit[qubit] = True
        except (ValueError, RuntimeError) as e:
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
