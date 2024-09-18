from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from scipy.optimize import curve_fit

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.config import log

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

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    qubit_frequency = {}
    bare_resonator_frequency = {}
    offset = {}
    matrix_element = {}
    charging_energy = {}
    for qubit in targets:
        qubit_frequency[qubit] = platform.qubits[qubit].drive_frequency
        bare_resonator_frequency[qubit] = platform.qubits[
            qubit
        ].bare_resonator_frequency
        matrix_element[qubit] = platform.qubits[qubit].crosstalk_matrix[qubit]
        offset[qubit] = -platform.qubits[qubit].sweetspot * matrix_element[qubit]
        charging_energy[qubit] = -platform.qubits[qubit].anharmonicity
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulses[qubit])

    # define the parameters to sweep and their range:
    delta_frequency_range = np.arange(
        -params.freq_width / 2, params.freq_width / 2, params.freq_step
    )
    freq_sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        [ro_pulses[qubit] for qubit in targets],
        type=SweeperType.OFFSET,
    )

    delta_bias_range = np.arange(
        -params.bias_width / 2, params.bias_width / 2, params.bias_step
    )
    sweepers = [
        Sweeper(
            Parameter.bias,
            delta_bias_range,
            qubits=[platform.qubits[qubit] for qubit in targets],
            type=SweeperType.OFFSET,
        )
    ]

    data = ResonatorFluxData(
        resonator_type=platform.resonator_type,
        qubit_frequency=qubit_frequency,
        offset=offset,
        matrix_element=matrix_element,
        bare_resonator_frequency=bare_resonator_frequency,
        charging_energy=charging_energy,
    )
    options = ExecutionParameters(
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    for bias_sweeper in sweepers:
        results = platform.sweep(sequence, options, bias_sweeper, freq_sweeper)
        # retrieve the results for every qubit
        for qubit in targets:
            result = results[ro_pulses[qubit].serial]
            sweetspot = platform.qubits[qubit].sweetspot
            data.register_qubit(
                qubit,
                signal=result.magnitude,
                phase=result.phase,
                freq=delta_frequency_range + ro_pulses[qubit].frequency,
                bias=delta_bias_range + sweetspot,
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
            coupling[qubit] = popt[0]
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
    update.bare_resonator_frequency(results.bare_resonator_freq[qubit], platform, qubit)
    update.readout_frequency(results.resonator_freq[qubit], platform, qubit)
    update.coupling(results.coupling[qubit], platform, qubit)


resonator_flux = Routine(_acquisition, _fit, _plot, _update)
"""ResonatorFlux Routine object."""
