from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Union

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

from ..utils import GHZ_TO_HZ, HZ_TO_GHZ, table_dict, table_html
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
    flux_amplitude_start: Optional[Union[int, float, List[float]]] = None
    """Amplitude start value(s) for flux pulses sweep relative to the qubit sweetspot [a.u.]."""
    flux_amplitude_end: Optional[Union[int, float, List[float]]] = None
    """Amplitude end value(s) for flux pulses sweep relative to the qubit sweetspot [a.u.]."""
    flux_amplitude_step: Optional[Union[int, float, List[float]]] = None
    """Amplitude step(s) for flux pulses sweep [a.u.]."""

    def __post_init__(self):
        if not self.has_bias_params:
            if self.has_flux_params:
                self.check_flux_params()
                return
        if not self.has_flux_params:
            if self.has_bias_params:
                return
        raise ValueError(
            "Too many arguments provided. Provide either bias_width "
            "and bias_step or flux_amplitude_width and flux_amplitude_step."
        )

    def check_flux_params(self):
        """All flux params must be either all float or all lists with the same length.

        This function does not check if the lenght of the lists is equal to the number
        of qubits in the experiment.
        """
        flux_params = (
            self.flux_amplitude_start,
            self.flux_amplitude_end,
            self.flux_amplitude_step,
        )
        if all(isinstance(param, (int, float)) for param in flux_params):
            return

        if all(isinstance(param, list) for param in flux_params):
            if all(len(param) == len(flux_params[0]) for param in flux_params):
                return
            raise ValueError("Flux lists do not have the same length.")
        raise ValueError(
            "flux parameters have the wrong type. Expected one of (int, float, list)."
        )

    @property
    def has_bias_params(self):
        """True if both bias_width and bias_step are set."""
        return self.bias_width is not None and self.bias_step is not None

    @property
    def has_flux_params(self):
        """True if both all flux amplitude parameters are set."""
        return (
            self.flux_amplitude_start is not None
            and self.flux_amplitude_end is not None
            and self.flux_amplitude_step is not None
        )

    @property
    def flux_pulses(self):
        """True if sweeping flux pulses, False if sweeping bias."""
        if self.has_flux_params:
            return True
        return False


@dataclass
class ResonatorFluxResults(Results):
    """ResonatoFlux outputs."""

    frequency: dict[QubitId, float] = field(default_factory=dict)
    """Readout frequency for each qubit."""
    sweetspot: dict[QubitId, float] = field(default_factory=dict)
    """Sweetspot for each qubit."""
    asymmetry: dict[QubitId, float] = field(default_factory=dict)
    """Asymmetry between junctions."""
    bare_frequency: dict[QubitId, float] = field(default_factory=dict)
    """Resonator bare frequency."""
    drive_frequency: dict[QubitId, float] = field(default_factory=dict)
    """Qubit frequency at sweetspot."""
    fitted_parameters: dict[QubitId, dict[str, float]] = field(default_factory=dict)
    """Raw fitting output."""
    coupling: dict[QubitId, float] = field(default_factory=dict)
    """Qubit-resonator coupling."""
    matrix_element: dict[QubitId, float] = field(default_factory=dict)
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

    resonator_type: str
    """Resonator type."""
    flux_pulses: bool
    """True if sweeping flux pulses, False if sweeping bias."""
    qubit_frequency: dict[QubitId, float] = field(default_factory=dict)
    """Qubit frequencies."""
    offset: dict[QubitId, float] = field(default_factory=dict)
    """Qubit bias offset."""
    bare_resonator_frequency: dict[QubitId, int] = field(default_factory=dict)
    """Qubit bare resonator frequency power provided by the user."""

    data: dict[QubitId, npt.NDArray[ResFluxType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, freq, bias, signal, phase):
        """Store output for single qubit."""
        self.data[qubit] = utils.create_data_array(
            freq, bias, signal, phase, dtype=ResFluxType
        )


def create_flux_pulse_sweepers(
    params: ResonatorFluxParameters,
    platform: Platform,
    qubits: list[QubitId],
    sequence: PulseSequence,
    crosstalk: bool = False,
) -> tuple[np.ndarray, list[Sweeper], list[PulseSequence]]:
    """Create a list of sweepers containing flux pulses.

    Args:
        params (ResonatorFluxParameters): parameters of the experiment (here flux amplitude is used).
        platform (Platform): platform on which to run the experiment.
        qubits (Qubits): qubits on which to run the experiment.
        sequence (PulseSequence): pulse sequence of the experiment (updated with flux pulses).
        crosstalk (bool): if True it will split amplitude sweepers (necessary for crosstalk protocol)
    """
    qf_pulses = {}
    sequences = [deepcopy(sequence) for _ in range(len(qubits))]
    for i, qubit in enumerate(qubits):
        if isinstance(params.flux_amplitude_start, list):
            flux_amplitude_start = params.flux_amplitude_start[i]
            flux_amplitude_end = params.flux_amplitude_end[i]
            flux_amplitude_step = params.flux_amplitude_step[i]
        else:
            flux_amplitude_start = params.flux_amplitude_start
            flux_amplitude_end = params.flux_amplitude_end
            flux_amplitude_step = params.flux_amplitude_step
        delta_bias_flux_range = np.arange(
            flux_amplitude_start,
            flux_amplitude_end,
            flux_amplitude_step,
        )
        pulse = platform.create_qubit_flux_pulse(
            qubit,
            start=0,
            duration=sequence.duration,
        )
        qf_pulses[qubit] = pulse
        if crosstalk:
            sequences[i].add(pulse)
        else:
            sequence.add(pulse)

    if crosstalk:
        sweepers = [
            Sweeper(
                Parameter.amplitude,
                delta_bias_flux_range,
                pulses=[qf_pulses[qubit]],
                type=SweeperType.ABSOLUTE,
            )
            for qubit in qubits
        ]
        return delta_bias_flux_range, sweepers, sequences
    else:
        sweepers = [
            Sweeper(
                Parameter.amplitude,
                delta_bias_flux_range,
                pulses=[qf_pulses[qubit] for qubit in qubits],
                type=SweeperType.ABSOLUTE,
            )
        ]
        return delta_bias_flux_range, sweepers, [sequence]


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
    for qubit in targets:
        qubit_frequency[qubit] = platform.qubits[qubit].drive_frequency
        bare_resonator_frequency[qubit] = platform.qubits[
            qubit
        ].bare_resonator_frequency
        offset[qubit] = platform.qubits[qubit].sweetspot
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulses[qubit])

    # define the parameters to sweep and their range:
    delta_frequency_range = np.arange(
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )
    freq_sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        [ro_pulses[qubit] for qubit in targets],
        type=SweeperType.OFFSET,
    )
    if params.flux_pulses:
        delta_bias_flux_range, sweepers, sequences = create_flux_pulse_sweepers(
            params, platform, targets, sequence
        )
        sequence = sequences[0]
    else:
        delta_bias_flux_range = np.arange(
            -params.bias_width / 2, params.bias_width / 2, params.bias_step
        )
        sweepers = [
            Sweeper(
                Parameter.bias,
                delta_bias_flux_range,
                qubits=[platform.qubits[qubit] for qubit in targets],
                type=SweeperType.OFFSET,
            )
        ]

    data = ResonatorFluxData(
        resonator_type=platform.resonator_type,
        flux_pulses=params.flux_pulses,
        qubit_frequency=qubit_frequency,
        offset=offset,
        bare_resonator_frequency=bare_resonator_frequency,
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
                bias=delta_bias_flux_range + sweetspot,
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
                utils.transmon_readout_frequency_diagonal,
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
                utils.transmon_readout_frequency_diagonal(popt[3], *popt) * GHZ_TO_HZ
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


def _plot(data: ResonatorFluxData, fit: ResonatorFluxResults, target: QubitId):
    """Plotting function for ResonatorFlux Experiment."""
    figures = utils.flux_dependence_plot(
        data, fit, target, utils.transmon_readout_frequency_diagonal
    )
    if data.flux_pulses:
        bias_flux_unit = "a.u."
    else:
        bias_flux_unit = "V"
    if fit is not None:
        fitting_report = table_html(
            table_dict(
                target,
                [
                    f"Sweetspot [{bias_flux_unit}]",
                    "Bare Resonator Frequency [Hz]",
                    "Readout Frequency [Hz]",
                    "Qubit Frequency at Sweetspot [Hz]",
                    "Asymmetry d",
                    "Coupling g",
                    "Flux dependence",
                ],
                [
                    np.round(fit.sweetspot[target], 4),
                    np.round(fit.bare_frequency[target], 4),
                    np.round(fit.frequency[target], 4),
                    np.round(fit.drive_frequency[target], 4),
                    np.round(fit.asymmetry[target], 4),
                    np.round(fit.coupling[target], 4),
                    np.round(fit.matrix_element[target], 4),
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
