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
from qibocal.auto.operation import Routine
from qibocal.config import log

from ..qubit_spectroscopy_ef import DEFAULT_ANHARMONICITY
from ..utils import HZ_TO_GHZ, extract_feature, table_dict, table_html
from . import utils
from .qubit_flux_dependence import (
    QubitFluxData,
    QubitFluxParameters,
    QubitFluxResults,
    QubitFluxType,
)
from .qubit_flux_dependence import _fit as diagonal_fit


@dataclass
class QubitCrosstalkParameters(QubitFluxParameters):
    """Crosstalk runcard inputs."""

    bias_point: Optional[dict[QubitId, float]] = field(default_factory=dict)
    """Dictionary with {qubit_id: bias_point_qubit_id}."""
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

    matrix_element: dict[QubitId, float] = field(default_factory=dict)
    """Diagonal flux element."""
    bias_point: dict[QubitId, float] = field(default_factory=dict)
    """Bias point for each qubit."""
    offset: dict[QubitId, float] = field(default_factory=dict)
    """Phase shift for each qubit."""
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

    @property
    def diagonal(self) -> Optional[QubitFluxData]:
        instance = QubitFluxData(
            resonator_type=self.resonator_type,
            charging_energy=self.charging_energy,
            qubit_frequency=self.qubit_frequency,
        )
        for qubit in self.qubits:
            try:
                instance.data[qubit] = self.data[qubit, qubit]
            except KeyError:
                log.info(
                    f"Diagonal acquisition not found for qubit {qubit}. Runcard values will be used to perform the off-diagonal fit."
                )

        if len(instance.data) > 0:
            return instance
        return QubitFluxData(
            resonator_type=self.resonator_type,
            charging_energy=self.charging_energy,
        )


@dataclass
class QubitCrosstalkResults(QubitFluxResults):
    """
    Qubit Crosstalk outputs.
    """

    qubit_frequency_bias_point: dict[QubitId, float]
    """Expected qubit frequency at bias point."""
    crosstalk_matrix: dict[QubitId, dict[QubitId, float]]
    """Crosstalk matrix element."""
    fitted_parameters: dict[tuple[QubitId, QubitId], dict]
    """Fitted parameters for each couple target-flux qubit."""

    def __contains__(self, key: QubitId):
        """Checking if qubit is in crosstalk_matrix attribute."""
        return key in self.crosstalk_matrix


def _acquisition(
    params: QubitCrosstalkParameters,
    platform: Platform,
    targets: list[QubitId],
) -> QubitCrosstalkData:
    """Data acquisition for Crosstalk Experiment."""

    # TODO: pass voltage as parameter
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    offset = {}
    charging_energy = {}
    matrix_element = {}
    qubit_frequency = {}
    bias_point = {}
    for qubit in targets:

        charging_energy[qubit] = -platform.qubits[qubit].anharmonicity
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=0, duration=params.drive_duration
        )
        try:
            qubit_frequency[qubit] = platform.qubits[qubit].drive_frequency
            matrix_element[qubit] = platform.qubits[qubit].crosstalk_matrix[qubit]
            offset[qubit] = -platform.qubits[qubit].sweetspot * matrix_element[qubit]
            bias_point[qubit] = params.bias_point.get(
                qubit, platform.qubits[qubit].sweetspot
            )

        except KeyError:
            log.warning(f"Missing flux parameters for qubit {qubit}.")

        if params.transition == "02":
            if platform.qubits[qubit].anharmonicity:
                qd_pulses[qubit].frequency -= platform.qubits[qubit].anharmonicity / 2
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
        -params.freq_width / 2, params.freq_width / 2, params.freq_step
    )
    freq_sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[qd_pulses[qubit] for qubit in targets],
        type=SweeperType.OFFSET,
    )
    # TODO : abstract common lines with qubit flux dep routine
    if params.flux_qubits is None:
        flux_qubits = list(platform.qubits)
    else:
        flux_qubits = params.flux_qubits

    delta_bias_range = np.arange(
        -params.bias_width / 2, params.bias_width / 2, params.bias_step
    )
    sequences = [sequence] * len(flux_qubits)
    sweepers = [
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
        matrix_element=matrix_element,
        offset=offset,
        qubit_frequency=qubit_frequency,
        charging_energy=charging_energy,
        bias_point=bias_point,
    )

    options = ExecutionParameters(
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    # update bias configuration
    for qubit in targets:
        if qubit in params.bias_point:
            platform.qubits[qubit].flux.offset = params.bias_point[qubit]

    for flux_qubit, bias_sweeper, sequence in zip(flux_qubits, sweepers, sequences):
        results = platform.sweep(sequence, options, bias_sweeper, freq_sweeper)

        # retrieve the results for every qubit
        for qubit in targets:
            result = results[ro_pulses[qubit].serial]
            if flux_qubit is None:
                sweetspot = platform.qubits[qubit].flux.offset
            else:
                sweetspot = platform.qubits[flux_qubit].flux.offset
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

    crosstalk_matrix = {qubit: {} for qubit in data.qubit_frequency}
    fitted_parameters = {}

    bias_point = {}
    sweetspot = {}
    matrix_element = {}
    qubit_frequency = {}
    qubit_frequency_bias_point = {}
    offset = {}

    diagonal = diagonal_fit(data.diagonal)
    for qubit in data.qubits:
        condition = qubit in diagonal
        bias_point[qubit] = data.bias_point[qubit]
        matrix_element[qubit] = (
            diagonal.matrix_element[qubit] if condition else data.matrix_element[qubit]
        )
        qubit_frequency[qubit] = (
            diagonal.frequency[qubit] if condition else data.qubit_frequency[qubit]
        )
        offset[qubit] = (
            -diagonal.sweetspot[qubit] * diagonal.matrix_element[qubit]
            if condition
            else data.offset[qubit]
        )

    for target_flux_qubit, qubit_data in data.data.items():
        frequencies, biases = extract_feature(
            qubit_data.freq,
            qubit_data.bias,
            qubit_data.signal,
            "max" if data.resonator_type == "2D" else "min",
        )
        target_qubit, flux_qubit = target_flux_qubit

        if target_qubit != flux_qubit:
            # fit function needs to be defined here to pass correct parameters
            # at runtime
            qubit_frequency_bias_point[target_qubit] = utils.transmon_frequency(
                xi=bias_point[target_qubit],
                xj=0,
                d=0,
                w_max=qubit_frequency[target_qubit] * HZ_TO_GHZ,
                offset=offset[qubit],
                normalization=matrix_element[target_qubit],
                charging_energy=data.charging_energy[target_qubit] * HZ_TO_GHZ,
                crosstalk_element=1,
            )

            def fit_function(x, crosstalk_element, offset):
                return utils.transmon_frequency(
                    xi=bias_point[target_qubit],
                    xj=x,
                    d=0,
                    w_max=qubit_frequency[target_qubit] * HZ_TO_GHZ,
                    offset=offset,
                    normalization=matrix_element[target_qubit],
                    charging_energy=data.charging_energy[target_qubit] * HZ_TO_GHZ,
                    crosstalk_element=crosstalk_element,
                )

            try:
                popt, _ = curve_fit(
                    fit_function,
                    biases,
                    frequencies * HZ_TO_GHZ,
                    bounds=((-np.inf, -1), (np.inf, 1)),
                    maxfev=100000,
                )
                fitted_parameters[target_qubit, flux_qubit] = dict(
                    xi=bias_point[target_qubit],
                    d=0,
                    w_max=qubit_frequency[target_qubit] * HZ_TO_GHZ,
                    offset=popt[1],
                    normalization=matrix_element[target_qubit],
                    charging_energy=data.charging_energy[target_qubit] * HZ_TO_GHZ,
                    crosstalk_element=float(popt[0]),
                )
                crosstalk_matrix[target_qubit][flux_qubit] = (
                    popt[0] * matrix_element[target_qubit]
                )
            except RuntimeError as e:
                log.error(
                    f"Off-diagonal flux fit failed for qubit {flux_qubit} due to {e}."
                )

        else:
            fitted_parameters[target_qubit, flux_qubit] = diagonal.fitted_parameters[
                target_qubit
            ]
            crosstalk_matrix[target_qubit][flux_qubit] = diagonal.matrix_element[
                target_qubit
            ]

    return QubitCrosstalkResults(
        frequency=qubit_frequency,
        qubit_frequency_bias_point=qubit_frequency_bias_point,
        sweetspot=sweetspot,
        matrix_element=matrix_element,
        crosstalk_matrix=crosstalk_matrix,
        fitted_parameters=fitted_parameters,
    )


def _plot(data: QubitCrosstalkData, fit: QubitCrosstalkResults, target: QubitId):
    """Plotting function for Crosstalk Experiment."""
    figures, fitting_report = utils.flux_crosstalk_plot(
        data, target, fit, fit_function=utils.transmon_frequency
    )
    if fit is not None:
        labels = [
            "Qubit Frequency at Sweetspot [Hz]",
            "Qubit Frequency at Bias point [Hz]",
        ]
        values = [
            np.round(fit.frequency[target], 4),
            np.round(fit.qubit_frequency_bias_point[target], 4),
        ]
        for flux_qubit in fit.crosstalk_matrix[target]:
            if flux_qubit != target:
                labels.append(f"Crosstalk with qubit {flux_qubit} [V^-1]")
            else:
                labels.append(f"Flux dependence [V^-1]")
            values.append(np.round(fit.crosstalk_matrix[target][flux_qubit], 4))
        fitting_report = table_html(
            table_dict(
                target,
                labels,
                values,
            )
        )
    return figures, fitting_report


def _update(results: QubitCrosstalkResults, platform: Platform, qubit: QubitId):
    """Update crosstalk matrix."""

    for flux_qubit, element in results.crosstalk_matrix[qubit].items():
        update.crosstalk_matrix(element, platform, qubit, flux_qubit)


qubit_crosstalk = Routine(_acquisition, _fit, _plot, _update)
"""Qubit crosstalk Routine object"""
