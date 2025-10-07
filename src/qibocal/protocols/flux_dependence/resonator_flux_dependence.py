from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, Parameter, PulseSequence, Sweeper
from scipy.optimize import curve_fit

from qibocal.calibration import CalibrationPlatform

from ... import update
from ...auto.operation import Data, QubitId, Results, Routine
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
from . import utils

__all__ = ["ResonatorFluxParameters", "resonator_flux"]


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
        return extract_feature(
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

    for qubit in data.qubits:
        # extract signal from 2D plot based on SNR mask
        frequencies, biases = data.filtered_data(qubit)

        # define fit function
        def fit_function(
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
            popt, _ = curve_fit(
                fit_function,
                biases,
                frequencies * HZ_TO_GHZ,
                bounds=(
                    [
                        0,
                        0,
                        -1,
                        0,
                        data.bare_resonator_frequency[qubit] * HZ_TO_GHZ - 0.5,
                        0,
                    ],
                    [
                        0.5,
                        1,
                        1,
                        np.inf,
                        data.bare_resonator_frequency[qubit] * HZ_TO_GHZ + 0.5,
                        data.charging_energy[qubit] * HZ_TO_GHZ + 0.3,
                    ],
                ),
                maxfev=100000,
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
            resonator_freq[qubit] = fit_function(sweetspot[qubit], *popt) * GHZ_TO_HZ
            coupling[qubit] = popt[0]
            asymmetry[qubit] = popt[1]
        except ValueError as e:
            log.error(f"Error in resonator_flux protocol fit: {e} ")
    return ResonatorFluxResults(
        frequency=resonator_freq,
        coupling=coupling,
        matrix_element=matrix_element,
        sweetspot=sweetspot,
        asymmetry=asymmetry,
        fitted_parameters=fitted_parameters,
    )


def _plot(data: ResonatorFluxData, fit: ResonatorFluxResults, target: QubitId):
    """Plotting function for ResonatorFlux Experiment."""
    figures = utils.flux_dependence_plot(
        data, fit, target, utils.transmon_readout_frequency
    )

    if fit is not None:
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
    update.dressed_resonator_frequency(results.frequency[qubit], platform, qubit)
    update.readout_frequency(results.frequency[qubit], platform, qubit)
    update.coupling(results.coupling[qubit], platform, qubit)
    update.flux_offset(results.sweetspot[qubit], platform, qubit)
    update.sweetspot(results.sweetspot[qubit], platform, qubit)


resonator_flux = Routine(_acquisition, _fit, _plot, _update)
"""ResonatorFlux Routine object."""
