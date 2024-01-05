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
from qibocal.auto.operation import Qubits, Results, Routine

from ..qubit_spectroscopy_ef import DEFAULT_ANHARMONICITY
from ..utils import HZ_TO_GHZ
from . import utils
from .qubit_flux_dependence import QubitFluxData, QubitFluxParameters, QubitFluxType


@dataclass
class QubitCrosstalkParameters(QubitFluxParameters):
    """Crosstalk runcard inputs."""

    flux_qubits: Optional[list[QubitId]] = None
    """IDs of the qubits that we will sweep the flux on.
    If ``None`` flux will be swept on all qubits that we are running the routine on in a multiplex fashion.
    If given flux will be swept on the given qubits in a sequential fashion (n qubits will result to n different executions).
    Multiple qubits may be measured in each execution as specified by the ``qubits`` option in the runcard.
    """
    # TODO: add voltage parameters to bias qubits off sweetspot (absolute)


@dataclass
class QubitCrosstalkData(QubitFluxData):
    """Crosstalk acquisition outputs when ``flux_qubits`` are given."""

    sweetspot: dict[QubitId, float] = field(default_factory=dict)
    """Sweetspot for each qubit."""
    d: dict[QubitId, float] = field(default_factory=dict)
    """Asymmetry for each qubit."""
    voltage: dict[QubitId, float] = field(default_factory=dict)
    """Voltage provided to each qubit."""
    matrix_element: dict[QubitId, float] = field(default_factory=dict)
    """Diagonal crosstalk matrix element."""
    qubit_frequency: dict[QubitId, float] = field(default_factory=dict)
    """Qubit frequency for each qubit."""
    data: dict[tuple[QubitId, QubitId], npt.NDArray[QubitFluxType]] = field(
        default_factory=dict
    )
    """Raw data acquired for (qubit, qubit_flux) pairs saved in nested dictionaries."""

    def register_qubit(self, qubit, flux_qubit, freq, bias, signal, phase):
        """Store output for single qubit."""
        ar = utils.create_data_array(freq, bias, signal, phase, dtype=QubitFluxType)
        if (qubit, flux_qubit) in self.data:
            self.data[qubit, flux_qubit] = np.rec.array(
                np.concatenate((self.data[qubit, flux_qubit], ar))
            )
        else:
            self.data[qubit, flux_qubit] = ar


@dataclass
class QubitCrosstalkResults(Results):
    """
    Qubit Crosstalk outputs.
    """

    crosstalk_matrix: dict[QubitId, dict[QubitId, float]] = field(default_factory=dict)
    """Crosstalk matrix element."""
    fitted_parameters: dict[tuple[QubitId, QubitId], dict] = field(default_factory=dict)
    """Fitted parameters for each couple target-flux qubit."""


def _acquisition(
    params: QubitCrosstalkParameters,
    platform: Platform,
    qubits: Qubits,
) -> QubitCrosstalkData:
    """Data acquisition for Crosstalk Experiment."""

    # TODO: pass voltage as parameter
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    sweetspots = {}
    d = {}
    voltage = {}
    matrix_element = {}
    qubit_frequency = {}
    for qubit in qubits:
        sweetspots[qubit] = voltage[qubit] = platform.qubits[qubit].sweetspot
        d[qubit] = platform.qubits[qubit].asymmetry
        matrix_element[qubit] = platform.qubits[qubit].crosstalk_matrix[qubit]
        qubit_frequency[qubit] = platform.qubits[qubit].drive_frequency * HZ_TO_GHZ
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=0, duration=params.drive_duration
        )

        if params.transition == "02":
            if qubits[qubit].anharmonicity:
                qd_pulses[qubit].frequency -= qubits[qubit].anharmonicity / 2
            else:
                qd_pulses[qubit].frequency -= DEFAULT_ANHARMONICITY / 2

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
    # TODO : abstract common lines with qubit flux dep routine
    if params.flux_qubits is None:
        flux_qubits = list(platform.qubits.keys())
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
    data = QubitCrosstalkData(
        resonator_type=platform.resonator_type,
        sweetspot=sweetspots,
        voltage=voltage,
        matrix_element=matrix_element,
        d=d,
        qubit_frequency=qubit_frequency,
    )

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
                signal=result.magnitude,
                phase=result.phase,
                freq=delta_frequency_range + qd_pulses[qubit].frequency,
                bias=delta_bias_range + sweetspot,
            )

    return data


def _fit(data: QubitCrosstalkData) -> QubitCrosstalkResults:
    # FIXME: currently this method performs the fit ONLY of the off-diagonal
    # elements. An alternative should be to perform first the fit of the diagonal
    # elements and then use those parameters to perform the fit of the off-diagonal
    # elements.
    crosstalk_matrix = {qubit: {} for qubit in data.qubit_frequency}
    fitted_parameters = {}
    for target_flux_qubit, qubit_data in data.data.items():
        target_qubit, flux_qubit = target_flux_qubit

        if data.resonator_type == "3D":
            frequencies, biases = utils.extract_min_feature(
                qubit_data.freq,
                qubit_data.bias,
                qubit_data.signal,
            )
        else:
            frequencies, biases = utils.extract_max_feature(
                qubit_data.freq,
                qubit_data.bias,
                qubit_data.signal,
            )
        if target_qubit != flux_qubit:
            # fit function needs to be defined here to pass correct parameters
            # at runtime
            def fit_function(x, crosstalk_element):
                return utils.transmon_frequency(
                    xi=data.voltage[target_qubit],
                    xj=x,
                    w_max=data.qubit_frequency[target_qubit],
                    d=data.d[target_qubit],
                    sweetspot=data.sweetspot[target_qubit],
                    matrix_element=data.matrix_element[target_qubit],
                    crosstalk_element=crosstalk_element,
                )

            popt, _ = curve_fit(
                fit_function, biases, frequencies / 1e9, bounds=(0, np.inf)
            )
            fitted_parameters[target_qubit, flux_qubit] = dict(
                xi=data.voltage[target_qubit],
                w_max=data.qubit_frequency[target_qubit],
                d=data.d[target_qubit],
                sweetspot=data.sweetspot[target_qubit],
                matrix_element=data.matrix_element[target_qubit],
                crosstalk_element=float(popt),
            )
            crosstalk_matrix[target_qubit][flux_qubit] = 1 / float(popt)

    return QubitCrosstalkResults(
        crosstalk_matrix=crosstalk_matrix, fitted_parameters=fitted_parameters
    )


def _plot(data: QubitCrosstalkData, fit: QubitCrosstalkResults, qubit):
    """Plotting function for Crosstalk Experiment."""
    return utils.flux_crosstalk_plot(
        data, qubit, fit, fit_function=utils.transmon_frequency
    )


def _update(results: QubitCrosstalkResults, platform: Platform, qubit: QubitId):
    """Update crosstalk matrix."""
    for flux_qubit, element in results.crosstalk_matrix[qubit].items():
        update.crosstalk_matrix(element, platform, qubit, flux_qubit)


qubit_crosstalk = Routine(_acquisition, _fit, _plot, _update)
"""Qubit crosstalk Routine object"""
