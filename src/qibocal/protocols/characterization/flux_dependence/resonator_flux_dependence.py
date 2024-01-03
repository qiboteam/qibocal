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

from ..utils import GHZ_TO_HZ
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
    d: dict[QubitId, float]
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""


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
        resonator_type=platform.resonator_type, qubit_frequency=qubit_frequency
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
    d = {}

    fitted_parameters = {}

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

        popt = curve_fit(
            utils.transmon_readout_frequency,
            biases,
            frequencies / 1e9,
            bounds=(
                [
                    data.qubit_frequency[qubit] / 1e9 - 0.5,
                    0,
                    0,
                    np.mean(qubit_data.bias) - 0.5,
                    np.mean(qubit_data.freq) / 1e9 - 1,
                    0,
                ],
                [
                    data.qubit_frequency[qubit] / 1e9 + 0.5,
                    1,
                    np.inf,
                    np.mean(qubit_data.bias) + 0.5,
                    np.mean(qubit_data.freq) / 1e9 + 1,
                    1,
                ],
            ),
            maxfev=100000,
        )[0]
        fitted_parameters[qubit] = popt.tolist()

        # frequency corresponds to transmon readout frequency
        # at the sweetspot popt[3]
        frequency[qubit] = utils.transmon_readout_frequency(popt[3], *popt) * GHZ_TO_HZ
        sweetspot[qubit] = popt[3]
        d[qubit] = popt[1]

    return ResonatorFluxResults(
        frequency=frequency,
        sweetspot=sweetspot,
        d=d,
        fitted_parameters=fitted_parameters,
    )


def _plot(data: ResonatorFluxData, fit: ResonatorFluxResults, qubit):
    """Plotting function for ResonatorFlux Experiment."""
    return utils.flux_dependence_plot(data, fit, qubit)


def _update(results: ResonatorFluxResults, platform: Platform, qubit: QubitId):
    # update.bare_resonator_frequency_sweetspot(results.brf[qubit], platform, qubit)
    update.readout_frequency(results.frequency[qubit], platform, qubit)
    # update.flux_to_bias(results.flux_to_bias[qubit], platform, qubit)
    update.asymmetry(results.d[qubit], platform, qubit)
    # update.ratio_sweetspot_qubit_freq_bare_resonator_freq(
    #     results.ssf_brf[qubit], platform, qubit
    # )
    # update.charging_energy(results.ECs[qubit], platform, qubit)
    # update.josephson_energy(results.EJs[qubit], platform, qubit)
    # update.coupling(results.Gs[qubit], platform, qubit)


resonator_flux = Routine(_acquisition, _fit, _plot, _update)
"""ResonatorFlux Routine object."""
