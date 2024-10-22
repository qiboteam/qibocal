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

from ... import update
from ...auto.operation import Routine
from ...config import log
from ..utils import HZ_TO_GHZ, extract_feature, table_dict, table_html
from . import utils
from .resonator_flux_dependence import (
    ResFluxType,
    ResonatorFluxData,
    ResonatorFluxParameters,
    ResonatorFluxResults,
)
from .resonator_flux_dependence import _fit as diagonal_fit


@dataclass
class ResCrosstalkParameters(ResonatorFluxParameters):
    """ResonatorFlux runcard inputs."""

    bias_point: Optional[dict[QubitId, float]] = field(default_factory=dict)
    """Dictionary with {qubit_id: bias_point_qubit_id}."""
    flux_qubits: Optional[list[QubitId]] = None
    """IDs of the qubits that we will sweep the flux on.
    If ``None`` flux will be swept on all qubits that we are running the routine on in a multiplex fashion.
    If given flux will be swept on the given qubits in a sequential fashion (n qubits will result to n different executions).
    Multiple qubits may be measured in each execution as specified by the ``qubits`` option in the runcard.
    """


@dataclass
class ResCrosstalkResults(ResonatorFluxResults):
    """ResCrosstalk outputs."""

    resonator_frequency_bias_point: dict[QubitId, dict[QubitId, float]] = field(
        default_factory=dict
    )
    """Resonator frequency at bias point."""
    crosstalk_matrix: dict[QubitId, dict[QubitId, float]] = field(default_factory=dict)
    """Crosstalk matrix element."""
    fitted_parameters: dict[tuple[QubitId, QubitId], dict] = field(default_factory=dict)
    """Fitted parameters for each couple target-flux qubit."""

    def __contains__(self, key: QubitId):
        """Checking if qubit is in crosstalk_matrix attribute."""
        return key in self.crosstalk_matrix


@dataclass
class ResCrosstalkData(ResonatorFluxData):
    """ResFlux acquisition outputs when ``flux_qubits`` are given."""

    coupling: dict[QubitId, float] = field(default_factory=dict)
    """Coupling parameter g for each qubit."""
    bias_point: dict[QubitId, float] = field(default_factory=dict)
    """Voltage provided to each qubit."""
    bare_resonator_frequency: dict[QubitId, float] = field(default_factory=dict)
    """Readout resonator frequency for each qubit."""
    resonator_frequency: dict[QubitId, float] = field(default_factory=dict)
    """Readout resonator frequency for each qubit."""
    matrix_element: dict[QubitId, float] = field(default_factory=dict)
    """Diagonal crosstalk matrix element."""
    offset: dict[QubitId, float] = field(default_factory=dict)
    """Diagonal offset."""
    asymmetry: dict[QubitId, float] = field(default_factory=dict)
    """Diagonal asymmetry."""
    data: dict[tuple[QubitId, QubitId], npt.NDArray[ResFluxType]] = field(
        default_factory=dict
    )
    """Raw data acquired for (qubit, qubit_flux) pairs saved in nested dictionaries."""

    def register_qubit(self, qubit, flux_qubit, freq, bias, signal, phase):
        """Store output for single qubit."""
        ar = utils.create_data_array(freq, bias, signal, phase, dtype=ResFluxType)
        if (qubit, flux_qubit) in self.data:
            self.data[qubit, flux_qubit] = np.rec.array(
                np.concatenate((self.data[qubit, flux_qubit], ar))
            )
        else:
            self.data[qubit, flux_qubit] = ar

    @property
    def diagonal(self) -> ResonatorFluxData:
        """Returns diagonal data acquired."""
        instance = ResonatorFluxData(
            resonator_type=self.resonator_type,
            qubit_frequency=self.qubit_frequency,
            bare_resonator_frequency=self.bare_resonator_frequency,
            charging_energy=self.charging_energy,
        )
        for qubit in self.qubits:
            try:
                instance.data[qubit] = self.data[qubit, qubit]
            except KeyError:
                log.info(
                    f"Diagonal acquisition not found for qubit {qubit}. Runcard values will be used to perform the off-diagonal fit."
                )

        return instance


def _acquisition(
    params: ResCrosstalkParameters, platform: Platform, targets: list[QubitId]
) -> ResCrosstalkData:
    """Data acquisition for ResonatorFlux experiment."""
    sequence = PulseSequence()
    ro_pulses = {}
    bare_resonator_frequency = {}
    resonator_frequency = {}
    qubit_frequency = {}
    coupling = {}
    asymmetry = {}
    charging_energy = {}
    bias_point = {}
    offset = {}
    matrix_element = {}
    for qubit in targets:
        charging_energy[qubit] = -platform.qubits[qubit].anharmonicity
        bias_point[qubit] = params.bias_point.get(
            qubit, platform.qubits[qubit].sweetspot
        )
        coupling[qubit] = platform.qubits[qubit].g
        asymmetry[qubit] = platform.qubits[qubit].asymmetry
        matrix_element[qubit] = platform.qubits[qubit].crosstalk_matrix[qubit]
        offset[qubit] = -platform.qubits[qubit].sweetspot * matrix_element[qubit]
        bare_resonator_frequency[qubit] = platform.qubits[
            qubit
        ].bare_resonator_frequency
        qubit_frequency[qubit] = platform.qubits[qubit].drive_frequency
        resonator_frequency[qubit] = platform.qubits[qubit].readout_frequency
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
    data = ResCrosstalkData(
        resonator_type=platform.resonator_type,
        qubit_frequency=qubit_frequency,
        offset=offset,
        asymmetry=asymmetry,
        resonator_frequency=resonator_frequency,
        charging_energy=charging_energy,
        bias_point=bias_point,
        matrix_element=matrix_element,
        coupling=coupling,
        bare_resonator_frequency=bare_resonator_frequency,
    )
    options = ExecutionParameters(
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
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
                freq=delta_frequency_range + ro_pulses[qubit].frequency,
                bias=delta_bias_range + sweetspot,
            )
    return data


def _fit(data: ResCrosstalkData) -> ResCrosstalkResults:
    """ "PostProcessing for resonator crosstalk protocol."""

    # perform first fit where corresponding qubit is moved
    diagonal = diagonal_fit(data.diagonal)

    fitted_parameters = {}
    crosstalk_matrix = {qubit: {} for qubit in data.qubit_frequency}
    offset = {}
    coupling = {}
    matrix_element = {}
    asymmetry = {}
    resonator_frequency = {}
    resonator_frequency_bias_point = {}

    for qubit in data.qubits:

        # retrieve parameters from diagonal fit if performed
        condition = qubit in diagonal
        coupling[qubit] = (
            diagonal.coupling[qubit] if condition else data.coupling[qubit]
        )
        asymmetry[qubit] = (
            diagonal.asymmetry[qubit] if condition else data.asymmetry[qubit]
        )
        matrix_element[qubit] = (
            diagonal.matrix_element[qubit] if condition else data.matrix_element[qubit]
        )
        resonator_frequency[qubit] = (
            diagonal.resonator_freq[qubit]
            if condition
            else data.resonator_frequency[qubit]
        )
        offset[qubit] = (
            diagonal.fitted_parameters[qubit]["offset"]
            if condition
            else data.offset[qubit]
        )

    for target_flux_qubit, qubit_data in data.data.items():
        target_qubit, flux_qubit = target_flux_qubit
        frequencies, biases = extract_feature(
            qubit_data.freq,
            qubit_data.bias,
            qubit_data.signal,
            "min" if data.resonator_type == "2D" else "max",
        )

        # fit valid only for non-diagonal case
        # (the diagonal case was handled before)
        if target_qubit != flux_qubit:
            resonator_frequency_bias_point[target_qubit] = (
                utils.transmon_readout_frequency(
                    xi=data.bias_point[target_qubit],
                    xj=0,
                    d=asymmetry[target_qubit],
                    w_max=data.qubit_frequency[target_qubit] * HZ_TO_GHZ,
                    offset=data.offset[target_qubit],
                    normalization=matrix_element[target_qubit],
                    charging_energy=data.charging_energy[target_qubit] * HZ_TO_GHZ,
                    g=coupling[target_qubit],
                    resonator_freq=data.bare_resonator_frequency[target_qubit]
                    * HZ_TO_GHZ,
                    crosstalk_element=1,
                )
            )

            def fit_function(x, crosstalk_element):
                return utils.transmon_readout_frequency(
                    xi=data.bias_point[target_qubit],
                    xj=x,
                    d=0,
                    w_max=data.qubit_frequency[target_qubit] * HZ_TO_GHZ,
                    offset=offset[target_qubit],
                    normalization=data.matrix_element[target_qubit],
                    charging_energy=data.charging_energy[target_qubit] * HZ_TO_GHZ,
                    g=coupling[target_qubit],
                    resonator_freq=data.bare_resonator_frequency[target_qubit]
                    * HZ_TO_GHZ,
                    crosstalk_element=crosstalk_element,
                )

            try:
                popt, _ = curve_fit(
                    fit_function,
                    biases,
                    frequencies * HZ_TO_GHZ,
                    bounds=(-1, 1),
                )
                fitted_parameters[target_qubit, flux_qubit] = dict(
                    xi=data.bias_point[qubit],
                    d=asymmetry[qubit],
                    w_max=data.qubit_frequency[target_qubit] * HZ_TO_GHZ,
                    offset=offset[qubit],
                    normalization=data.matrix_element[target_qubit],
                    charging_energy=data.charging_energy[target_qubit] * HZ_TO_GHZ,
                    g=coupling[target_qubit],
                    resonator_freq=data.bare_resonator_frequency[target_qubit]
                    * HZ_TO_GHZ,
                    crosstalk_element=float(popt[0]),
                )
                crosstalk_matrix[target_qubit][flux_qubit] = (
                    popt[0] * data.matrix_element[target_qubit]
                )
            except (ValueError, RuntimeError) as e:
                log.error(
                    f"Off-diagonal flux fit failed for qubit {flux_qubit} due to {e}."
                )
        else:
            fitted_parameters[target_qubit, flux_qubit] = diagonal.fitted_parameters[
                target_qubit
            ]
            crosstalk_matrix[target_qubit][flux_qubit] = matrix_element[qubit]

    return ResCrosstalkResults(
        resonator_freq=resonator_frequency,
        asymmetry=asymmetry,
        resonator_frequency_bias_point=resonator_frequency_bias_point,
        coupling=coupling,
        crosstalk_matrix=crosstalk_matrix,
        fitted_parameters=fitted_parameters,
    )


def _plot(data: ResCrosstalkData, fit: ResCrosstalkResults, target: QubitId):
    """Plotting function for ResonatorFlux Experiment."""
    figures, fitting_report = utils.flux_crosstalk_plot(
        data, target, fit, fit_function=utils.transmon_readout_frequency
    )
    if fit is not None:
        labels = [
            "Resonator Frequency at Sweetspot [Hz]",
            "Coupling g [MHz]",
            "Asymmetry d",
            "Resonator Frequency at Bias point [Hz]",
        ]
        values = [
            np.round(fit.resonator_freq[target], 4),
            np.round(fit.coupling[target] * 1e3, 2),
            np.round(fit.asymmetry[target], 2),
            np.round(fit.resonator_frequency_bias_point[target], 4),
        ]
        for flux_qubit in fit.crosstalk_matrix[target]:
            if flux_qubit != target:
                labels.append(f"Crosstalk with qubit {flux_qubit}")
            else:
                labels.append(f"Flux dependence")
            values.append(np.round(fit.crosstalk_matrix[target][flux_qubit], 4))

        fitting_report = table_html(
            table_dict(
                target,
                labels,
                values,
            )
        )
    return figures, fitting_report


def _update(results: ResCrosstalkResults, platform: Platform, qubit: QubitId):
    """Update crosstalk matrix."""
    for flux_qubit, element in results.crosstalk_matrix[qubit].items():
        update.crosstalk_matrix(element, platform, qubit, flux_qubit)


resonator_crosstalk = Routine(_acquisition, _fit, _plot, _update)
"""Resonator crosstalk Routine object"""
