from dataclasses import dataclass, field
from functools import partial
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
from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.config import log

from ..utils import GHZ_TO_HZ, HZ_TO_GHZ
from . import utils
from .resonator_flux_dependence import _fit_crosstalk


@dataclass
class QubitFluxParameters(Parameters):
    """QubitFlux runcard inputs."""

    freq_width: int
    """Width for frequency sweep relative to the qubit frequency (Hz)."""
    freq_step: int
    """Frequency step for sweep [Hz]."""
    bias_width: float
    """Width for bias sweep [V]."""
    bias_step: float
    """Bias step for sweep (V)."""
    drive_amplitude: Optional[float] = None
    """Drive amplitude (optional). If defined, same amplitude will be used in all qubits.
    Otherwise the default amplitude defined on the platform runcard will be used"""
    flux_qubits: Optional[list[QubitId]] = None
    """IDs of the qubits that we will sweep the flux on.
    If ``None`` flux will be swept on all qubits that we are running the routine on in a multiplex fashion.
    If given flux will be swept on the given qubits in a sequential fashion (n qubits will result to n different executions).
    Multiple qubits may be measured in each execution as specified by the ``qubits`` option in the runcard.
    """
    transition: Optional[str] = "01"
    """Flux spectroscopy transition type ("01" or "02"). Default value is 01"""


@dataclass
class QubitFluxResults(Results):
    """QubitFlux outputs."""

    sweetspot: dict[QubitId, float]
    """Sweetspot for each qubit."""
    frequency: dict[QubitId, float]
    """Drive frequency for each qubit."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""


@dataclass
class FluxCrosstalkResults(Results):
    """Empty fitting outputs for cross talk because fitting is not implemented in this case."""


QubitFluxType = np.dtype(
    [
        ("freq", np.float64),
        ("bias", np.float64),
        ("msr", np.float64),
        ("phase", np.float64),
    ]
)
"""Custom dtype for resonator flux dependence."""


@dataclass
class QubitFluxData(Data):
    """QubitFlux acquisition outputs."""

    """Resonator type."""
    resonator_type: str

    """ResonatorFlux acquisition outputs."""
    Ec: dict[QubitId, int] = field(default_factory=dict)
    """Qubit Ec provided by the user."""

    Ej: dict[QubitId, int] = field(default_factory=dict)
    """Qubit Ej provided by the user."""

    data: dict[QubitId, npt.NDArray[QubitFluxType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, flux_qubit, freq, bias, msr, phase):
        """Store output for single qubit."""
        self.data[qubit] = utils.create_data_array(
            freq, bias, msr, phase, dtype=QubitFluxType
        )


@dataclass
class FluxCrosstalkData(QubitFluxData):
    """QubitFlux acquisition outputs when ``flux_qubits`` are given."""

    data: dict[QubitId, dict[QubitId, npt.NDArray[QubitFluxType]]] = field(
        default_factory=dict
    )
    """Raw data acquired for (qubit, qubit_flux) pairs saved in nested dictionaries."""

    def register_qubit(self, qubit, flux_qubit, freq, bias, msr, phase):
        """Store output for single qubit."""
        ar = utils.create_data_array(freq, bias, msr, phase, dtype=QubitFluxType)
        if (qubit, flux_qubit) in self.data:
            self.data[qubit, flux_qubit] = np.rec.array(
                np.concatenate((self.data[qubit, flux_qubit], ar))
            )
        else:
            self.data[qubit, flux_qubit] = ar


def _acquisition(
    params: QubitFluxParameters,
    platform: Platform,
    qubits: Qubits,
) -> QubitFluxData:
    """Data acquisition for QubitFlux Experiment."""

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    Ec = {}
    Ej = {}
    for qubit in qubits:
        Ec[qubit] = qubits[qubit].Ec
        Ej[qubit] = qubits[qubit].Ej

        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=0, duration=2000
        )

        if params.transition == "02":
            qd_pulses[qubit].frequency -= qubits[qubit].anharmonicity / 2

        if params.drive_amplitude is not None:
            qd_pulses[qubit].amplitude = params.drive_amplitude

        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameters to sweep and their range:
    delta_frequency_range = np.arange(
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )
    freq_sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[qd_pulses[qubit] for qubit in qubits],
        type=SweeperType.OFFSET,
    )

    delta_bias_range = np.arange(
        -params.bias_width / 2, params.bias_width / 2, params.bias_step
    )
    if params.flux_qubits is None:
        flux_qubits = [None]
        bias_sweepers = [
            Sweeper(
                Parameter.bias,
                delta_bias_range,
                qubits=list(qubits.values()),
                type=SweeperType.OFFSET,
            )
        ]
        data = QubitFluxData(resonator_type=platform.resonator_type, Ec=Ec, Ej=Ej)

    else:
        flux_qubits = params.flux_qubits
        bias_sweepers = [
            Sweeper(
                Parameter.bias,
                delta_bias_range,
                qubits=[platform.qubits[flux_qubit]],
                type=SweeperType.OFFSET,
            )
            for flux_qubit in flux_qubits
        ]
        data = FluxCrosstalkData(resonator_type=platform.resonator_type, Ec=Ec, Ej=Ej)

    options = ExecutionParameters(
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    for flux_qubit, bias_sweeper in zip(flux_qubits, bias_sweepers):
        results = platform.sweep(sequence, options, bias_sweeper, freq_sweeper)
        # retrieve the results for every qubit
        for qubit in qubits:
            result = results[ro_pulses[qubit].serial]
            if flux_qubit is None:
                sweetspot = qubits[qubit].sweetspot
            else:
                sweetspot = platform.qubits[flux_qubit].sweetspot
            data.register_qubit(
                qubit,
                flux_qubit,
                msr=result.magnitude,
                phase=result.phase,
                freq=delta_frequency_range + qd_pulses[qubit].frequency,
                bias=delta_bias_range + sweetspot,
            )

    return data


def _fit(data: QubitFluxData) -> QubitFluxResults:
    """
    Post-processing for QubitFlux Experiment. See arxiv:0703002
    Fit frequency as a function of current for the flux qubit spectroscopy
    data (QubitFluxData): data object with information on the feature response at each current point.
    """

    qubits = data.qubits
    frequency = {}
    sweetspot = {}
    fitted_parameters = {}

    for qubit in qubits:
        qubit_data = data[qubit]
        Ec = data.Ec[qubit]
        Ej = data.Ej[qubit]

        frequency[qubit] = 0
        sweetspot[qubit] = 0
        fitted_parameters[qubit] = {
            "Xi": 0,
            "d": 0,
            "Ec": 0,
            "Ej": 0,
            "f_q_offset": 0,
            "C_ii": 0,
        }

        biases = qubit_data.bias
        frequencies = qubit_data.freq
        msr = qubit_data.msr

        if data.resonator_type == "2D":
            msr = -msr

        frequencies, biases = utils.image_to_curve(
            frequencies, biases, msr, msr_mask=0.3
        )
        max_c = biases[np.argmax(frequencies)]
        min_c = biases[np.argmin(frequencies)]
        xi = 1 / (2 * abs(max_c - min_c))  # Convert bias to flux.

        # First order approximation: Ec and Ej NOT provided
        if Ec == 0 and Ej == 0:
            try:
                f_q_0 = np.max(
                    frequencies
                )  # Initial estimation for qubit frequency at sweet spot.
                popt = curve_fit(
                    utils.freq_q_transmon,
                    biases,
                    frequencies / GHZ_TO_HZ,
                    p0=[max_c, xi, 0, f_q_0 / GHZ_TO_HZ],
                    bounds=((-np.inf, 0, 0, 0), (np.inf, np.inf, np.inf, np.inf)),
                    maxfev=2000000,
                )[0]
                popt[3] *= GHZ_TO_HZ
                f_qs = popt[3]  # Qubit frequency at sweet spot.
                f_q_offset = utils.freq_q_transmon(
                    0, *popt
                )  # Qubit frequenct at zero current.
                C_ii = (f_qs - f_q_offset) / popt[
                    0
                ]  # Corresponding flux matrix element.

                frequency[qubit] = f_qs * HZ_TO_GHZ
                sweetspot[qubit] = popt[0]
                fitted_parameters[qubit] = {
                    "Xi": popt[1],
                    "d": abs(popt[2]),
                    "f_q_offset": f_q_offset,
                    "C_ii": C_ii,
                }
            except:
                log.warning(
                    "qubit_flux_fit: The first order approximation fitting was not succesful"
                )

        # Second order approximation: Ec and Ej provided
        else:
            try:
                freq_q_mathieu1 = partial(utils.freq_q_mathieu, p5=0.4999)
                popt = curve_fit(
                    freq_q_mathieu1,
                    biases,
                    frequencies / GHZ_TO_HZ,
                    p0=[max_c, xi, 0, Ec / GHZ_TO_HZ, Ej / GHZ_TO_HZ],
                    bounds=(
                        (-np.inf, 0, 0, 0, 0),
                        (np.inf, np.inf, np.inf, np.inf, np.inf),
                    ),
                    maxfev=2000000,
                )[0]
                popt[3] *= GHZ_TO_HZ
                popt[4] *= GHZ_TO_HZ
                f_qs = utils.freq_q_mathieu(
                    popt[0], *popt
                )  # Qubit frequency at sweet spot.
                f_q_offset = utils.freq_q_mathieu(
                    0, *popt
                )  # Qubit frequenct at zero current.
                C_ii = (f_qs - f_q_offset) / popt[
                    0
                ]  # Corresponding flux matrix element.

                frequency[qubit] = f_qs * HZ_TO_GHZ
                sweetspot[qubit] = popt[0]
                fitted_parameters[qubit] = {
                    "Xi": popt[1],
                    "d": abs(popt[2]),
                    "Ec": popt[3],
                    "Ej": popt[4],
                    "f_q_offset": f_q_offset,
                    "C_ii": C_ii,
                }
            except:
                log.warning(
                    "qubit_flux_fit: The second order approximation fitting was not succesful"
                )

    return QubitFluxResults(
        frequency=frequency,
        sweetspot=sweetspot,
        fitted_parameters=fitted_parameters,
    )


def _plot(data: QubitFluxData, fit: QubitFluxResults, qubit):
    """Plotting function for QubitFlux Experiment."""
    if utils.is_crosstalk(data):
        return utils.flux_crosstalk_plot(data, fit, qubit)
    return utils.flux_dependence_plot(data, fit, qubit)


def _update(results: QubitFluxResults, platform: Platform, qubit: QubitId):
    update.drive_frequency(results.frequency[qubit], platform, qubit)
    update.sweetspot(results.sweetspot[qubit], platform, qubit)


qubit_flux = Routine(_acquisition, _fit, _plot, _update)
"""QubitFlux Routine object."""
qubit_crosstalk = Routine(_acquisition, _fit_crosstalk, _plot)
"""Qubit crosstalk Routine object"""
