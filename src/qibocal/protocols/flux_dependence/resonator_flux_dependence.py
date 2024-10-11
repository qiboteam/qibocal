from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Parameter,
    Platform,
    PulseSequence,
    Sweeper,
)
from scipy.optimize import curve_fit

from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.config import log
from qibocal.result import magnitude, phase

from ..utils import GHZ_TO_HZ, HZ_TO_GHZ, extract_feature, table_dict, table_html
from . import utils


@dataclass
class ResonatorFluxParameters(Parameters):
    """ResonatorFlux runcard inputs."""

    freq_width: int
    """Width for frequency sweep relative to the readout frequency [Hz]."""
    freq_step: int
    """Frequency step for sweep [Hz]."""
    bias_width: Optional[float] = None
    """Width for bias sweep [V]."""
    bias_step: Optional[float] = None
    """Bias step for sweep [a.u.]."""


@dataclass
class ResonatorFluxResults(Results):
    """ResonatoFlux outputs."""

    resonator_freq: dict[QubitId, float] = field(default_factory=dict)
    bare_resonator_freq: dict[QubitId, float] = field(default_factory=dict)
    coupling: dict[QubitId, float] = field(default_factory=dict)
    """Qubit-resonator coupling."""
    fitted_parameters: dict[QubitId, float] = field(default_factory=dict)


ResFluxType = np.dtype(
    [
        ("freq", np.float64),
        ("bias", np.float64),
        ("signal", np.float64),
        ("phase", np.float64),
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
    offset: dict[QubitId, float] = field(default_factory=dict)
    """Qubit bias offset."""
    bare_resonator_frequency: dict[QubitId, int] = field(default_factory=dict)
    """Qubit bare resonator frequency power provided by the user."""
    matrix_element: dict[QubitId, float] = field(default_factory=dict)
    charging_energy: dict[QubitId, float] = field(default_factory=dict)

    data: dict[QubitId, npt.NDArray[ResFluxType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, freq, bias, signal, phase):
        """Store output for single qubit."""
        self.data[qubit] = utils.create_data_array(
            freq, bias, signal, phase, dtype=ResFluxType
        )


def _acquisition(
    params: ResonatorFluxParameters, platform: Platform, targets: list[QubitId]
) -> ResonatorFluxData:
    """Data acquisition for ResonatorFlux experiment."""
    # create a sequence of pulses for the experiment:
    # MZ

    delta_frequency_range = np.arange(
        -params.freq_width / 2, params.freq_width / 2, params.freq_step
    )
    delta_offset_range = np.arange(
        -params.bias_width / 2, params.bias_width / 2, params.bias_step
    )
    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    qubit_frequency = {}
    bare_resonator_frequency = {}
    offset = {}
    matrix_element = {}
    charging_energy = {}
    freq_sweepers = []
    offset_sweepers = []
    for q in targets:
        ro_sequence = platform.natives.single_qubit[q].MZ()
        ro_pulses[q] = ro_sequence[0][1]
        sequence += ro_sequence

        qubit = platform.qubits[q]
        offset0 = platform.config(qubit.flux).offset
        freq0 = platform.config(qubit.probe).frequency

        freq_sweepers.append(
            Sweeper(
                parameter=Parameter.frequency,
                values=freq0 + delta_frequency_range,
                channels=[qubit.probe],
            )
        )
        offset_sweepers.append(
            Sweeper(
                parameter=Parameter.offset,
                values=offset0 + delta_offset_range,
                channels=[qubit.flux],
            )
        )

        qubit_frequency[q] = platform.config(qubit.drive).frequency
        bare_resonator_frequency[q] = 0  # qubit.bare_resonator_frequency
        matrix_element[q] = 1  # qubit.crosstalk_matrix[q]
        offset[q] = -offset0 * matrix_element[q]
        charging_energy[q] = 0  # -qubit.anharmonicity

    data = ResonatorFluxData(
        resonator_type=platform.resonator_type,
        qubit_frequency=qubit_frequency,
        offset=offset,
        matrix_element=matrix_element,
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
            phase=phase(result),
            freq=freq_sweepers[i].values,
            bias=offset_sweepers[i].values,
        )
    return data


def _fit(data: ResonatorFluxData) -> ResonatorFluxResults:
    """
    Post-processing for QubitFlux Experiment. See arxiv:0703002
    Fit frequency as a function of current for the flux qubit spectroscopy
    data (QubitFluxData): data object with information on the feature response at each current point.
    """

    qubits = data.qubits
    coupling = {}
    resonator_freq = {}
    bare_resonator_freq = {}
    fitted_parameters = {}
    for qubit in qubits:
        qubit_data = data[qubit]
        biases = qubit_data.bias
        frequencies = qubit_data.freq
        signal = qubit_data.signal
        frequencies, biases = extract_feature(
            frequencies, biases, signal, "min" if data.resonator_type == "2D" else "max"
        )

        def fit_function(x, g, resonator_freq):
            return utils.transmon_readout_frequency(
                xi=x,
                w_max=data.qubit_frequency[qubit] * HZ_TO_GHZ,
                xj=0,
                d=0,
                normalization=data.matrix_element[qubit],
                offset=data.offset[qubit],
                crosstalk_element=1,
                charging_energy=data.charging_energy[qubit] * HZ_TO_GHZ,
                resonator_freq=resonator_freq,
                g=g,
            )

        try:
            popt, perr = curve_fit(
                fit_function,
                biases,
                frequencies * HZ_TO_GHZ,
                bounds=(
                    [0, data.bare_resonator_frequency[qubit] * HZ_TO_GHZ - 0.2],
                    [0.5, data.bare_resonator_frequency[qubit] * HZ_TO_GHZ + 0.2],
                ),
                maxfev=100000,
            )

            fitted_parameters[qubit] = {
                "w_max": data.qubit_frequency[qubit] * HZ_TO_GHZ,
                "xj": 0,
                "d": 0,
                "normalization": data.matrix_element[qubit],
                "offset": data.offset[qubit],
                "crosstalk_element": 1,
                "charging_energy": data.charging_energy[qubit] * HZ_TO_GHZ,
                "resonator_freq": popt[1],
                "g": popt[0],
            }
            sweetspot = -data.offset[qubit] / data.matrix_element[qubit]
            resonator_freq[qubit] = fit_function(sweetspot, *popt) * GHZ_TO_HZ
            coupling[qubit] = popt[0] * GHZ_TO_HZ
            bare_resonator_freq[qubit] = popt[1] * GHZ_TO_HZ
        except ValueError as e:
            log.error(
                f"Error in resonator_flux protocol fit: {e} "
                "The threshold for the SNR mask is probably too high. "
                "Lowering the value of `threshold` in `extract_*_feature`"
                "should fix the problem."
            )
    return ResonatorFluxResults(
        resonator_freq=resonator_freq,
        bare_resonator_freq=bare_resonator_freq,
        coupling=coupling,
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
                    "Bare resonator freq [Hz]",
                    "Dressed resonator freq [Hz]",
                    "Chi [MHz]",
                ],
                [
                    np.round(fit.coupling[target] * 1e3, 2),
                    np.round(fit.bare_resonator_freq[target], 6),
                    np.round(fit.resonator_freq[target], 6),
                    np.round(
                        (fit.bare_resonator_freq[target] - fit.resonator_freq[target])
                        * 1e-6,
                        2,
                    ),
                ],
            )
        )
        return figures, fitting_report
    return figures, ""


def _update(results: ResonatorFluxResults, platform: Platform, qubit: QubitId):
    pass
    # update.bare_resonator_frequency(results.bare_resonator_freq[qubit], platform, qubit)
    # update.readout_frequency(results.resonator_freq[qubit], platform, qubit)
    # update.coupling(results.coupling[qubit], platform, qubit)


resonator_flux = Routine(_acquisition, _fit, _plot, _update)
"""ResonatorFlux Routine object."""
