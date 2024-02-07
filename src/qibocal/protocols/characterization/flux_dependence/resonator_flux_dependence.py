from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from scipy.optimize import curve_fit

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.config import log

from ..utils import GHZ_TO_HZ, HZ_TO_GHZ, table_dict, table_html
from . import utils


@dataclass
class ResonatorFluxParameters(Parameters):
    """ResonatorFlux runcard inputs."""

    freq_width: int
    """Width for frequency sweep relative to the readout frequency [Hz]."""
    freq_step: int
    """Frequency step for sweep [Hz]."""
    bias_width: float
    """Width for bias sweep [V]."""
    bias_step: float
    """Bias step for sweep [a.u.]."""


@dataclass
class ResonatorFluxResults(Results):
    """ResonatoFlux outputs."""

    frequency: dict[QubitId, float]
    """Readout frequency for each qubit."""
    sweetspot: dict[QubitId, float]
    """Sweetspot for each qubit."""
    asymmetry: dict[QubitId, float]
    """Asymmetry between junctions."""
    bare_frequency: dict[QubitId, float]
    """Resonator bare frequency."""
    drive_frequency: dict[QubitId, float]
    """Qubit frequency at sweetspot."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""
    coupling: dict[QubitId, float]
    """Qubit-resonator coupling."""
    matrix_element: dict[QubitId, float]
    """C_ii coefficient."""


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

    """Resonator type."""
    resonator_type: str

    qubit_frequency: dict[QubitId, float] = field(default_factory=dict)
    """Qubit frequencies."""

    bare_resonator_frequency: dict[QubitId, int] = field(default_factory=dict)
    """Qubit bare resonator frequency power provided by the user."""

    data: dict[QubitId, npt.NDArray[ResFluxType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, freq, bias, signal, phase):
        """Store output for single qubit."""
        self.data[qubit] = utils.create_data_array(
            freq, bias, signal, phase, dtype=ResFluxType
        )


def _acquisition(
    params: ResonatorFluxParameters, platform: Platform, qubits: Qubits
) -> ResonatorFluxData:
    """Data acquisition for ResonatorFlux experiment."""
    # create a sequence of pulses for the experiment:
    # MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    qubit_frequency = {}
    bare_resonator_frequency = {}
    for qubit in qubits:
        qubit_frequency[qubit] = platform.qubits[qubit].drive_frequency
        bare_resonator_frequency[qubit] = platform.qubits[
            qubit
        ].bare_resonator_frequency

        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulses[qubit])

    # define the parameters to sweep and their range:
    delta_frequency_range = np.arange(
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )
    freq_sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        [ro_pulses[qubit] for qubit in qubits],
        type=SweeperType.OFFSET,
    )

    delta_bias_range = np.arange(
        -params.bias_width / 2, params.bias_width / 2, params.bias_step
    )
    bias_sweepers = [
        Sweeper(
            Parameter.bias,
            delta_bias_range,
            qubits=list(qubits.values()),
            type=SweeperType.OFFSET,
        )
    ]

    data = ResonatorFluxData(
        resonator_type=platform.resonator_type,
        qubit_frequency=qubit_frequency,
        bare_resonator_frequency=bare_resonator_frequency,
    )

    options = ExecutionParameters(
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    for bias_sweeper in bias_sweepers:
        results = platform.sweep(sequence, options, bias_sweeper, freq_sweeper)
        # retrieve the results for every qubit
        for qubit in qubits:
            result = results[ro_pulses[qubit].serial]
            sweetspot = qubits[qubit].sweetspot
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
    frequency = {}
    sweetspot = {}
    asymmetry = {}
    bare_frequency = {}
    drive_frequency = {}
    fitted_parameters = {}
    matrix_element = {}
    coupling = {}

    for qubit in qubits:
        qubit_data = data[qubit]

        biases = qubit_data.bias
        frequencies = qubit_data.freq
        signal = qubit_data.signal

        if data.resonator_type == "3D":
            frequencies, biases = utils.extract_max_feature(
                frequencies,
                biases,
                signal,
            )
        else:
            frequencies, biases = utils.extract_min_feature(
                frequencies,
                biases,
                signal,
            )

        try:
            popt = curve_fit(
                utils.transmon_readout_frequency,
                biases,
                frequencies * HZ_TO_GHZ,
                bounds=utils.resonator_flux_dependence_fit_bounds(
                    data.qubit_frequency[qubit],
                    qubit_data.bias,
                    data.bare_resonator_frequency[qubit],
                ),
                maxfev=100000,
            )[0]
            fitted_parameters[qubit] = popt.tolist()

            # frequency corresponds to transmon readout frequency
            # at the sweetspot popt[3]
            frequency[qubit] = (
                utils.transmon_readout_frequency(popt[3], *popt) * GHZ_TO_HZ
            )
            sweetspot[qubit] = popt[3]
            asymmetry[qubit] = popt[1]
            bare_frequency[qubit] = popt[4] * GHZ_TO_HZ
            drive_frequency[qubit] = popt[0] * GHZ_TO_HZ
            coupling[qubit] = popt[5]
            matrix_element[qubit] = popt[2]
        except ValueError as e:
            log.error(
                f"Error in resonator_flux protocol fit: {e} "
                "The threshold for the SNR mask is probably too high. "
                "Lowering the value of `threshold` in `extract_*_feature`"
                "should fix the problem."
            )

    return ResonatorFluxResults(
        frequency=frequency,
        sweetspot=sweetspot,
        asymmetry=asymmetry,
        bare_frequency=bare_frequency,
        drive_frequency=drive_frequency,
        coupling=coupling,
        matrix_element=matrix_element,
        fitted_parameters=fitted_parameters,
    )


def _plot(data: ResonatorFluxData, fit: ResonatorFluxResults, qubit):
    """Plotting function for ResonatorFlux Experiment."""
    figures = utils.flux_dependence_plot(
        data, fit, qubit, utils.transmon_readout_frequency
    )
    if fit is not None:
        fitting_report = table_html(
            table_dict(
                qubit,
                [
                    "Sweetspot [V]",
                    "Bare Resonator Frequency [Hz]",
                    "Readout Frequency [Hz]",
                    "Qubit Frequency at Sweetspot [Hz]",
                    "Asymmetry d",
                    "Coupling g",
                    "V_ii [V]",
                ],
                [
                    np.round(fit.sweetspot[qubit], 4),
                    np.round(fit.bare_frequency[qubit], 4),
                    np.round(fit.frequency[qubit], 4),
                    np.round(fit.drive_frequency[qubit], 4),
                    np.round(fit.asymmetry[qubit], 4),
                    np.round(fit.coupling[qubit], 4),
                    np.round(fit.matrix_element[qubit], 4),
                ],
            )
        )
        return figures, fitting_report
    return figures, ""


def _update(results: ResonatorFluxResults, platform: Platform, qubit: QubitId):
    update.bare_resonator_frequency(results.bare_frequency[qubit], platform, qubit)
    update.readout_frequency(results.frequency[qubit], platform, qubit)
    update.drive_frequency(results.drive_frequency[qubit], platform, qubit)
    update.asymmetry(results.asymmetry[qubit], platform, qubit)
    update.coupling(results.coupling[qubit], platform, qubit)


resonator_flux = Routine(_acquisition, _fit, _plot, _update)
"""ResonatorFlux Routine object."""
