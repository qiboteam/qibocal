from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    PulseSequence,
    Sweeper,
)
from scipy.optimize import curve_fit

from qibocal import update
from qibocal.auto.operation import QubitId, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log

from ...result import magnitude
from ...update import replace
from ..utils import (
    GHZ_TO_HZ,
    HZ_TO_GHZ,
    extract_feature,
    readout_frequency,
    table_dict,
    table_html,
)
from . import utils
from .qubit_flux_dependence import (
    QubitFluxData,
    QubitFluxParameters,
    QubitFluxResults,
    QubitFluxType,
)

__all__ = ["qubit_crosstalk"]


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

    def register_qubit(self, qubit, flux_qubit, freq, bias, signal):
        """Store output for single qubit."""
        ar = utils.create_data_array(freq, bias, signal, dtype=QubitFluxType)
        if (qubit, flux_qubit) in self.data:
            self.data[qubit, flux_qubit] = np.rec.array(
                np.concatenate((self.data[qubit, flux_qubit], ar))
            )
        else:
            self.data[qubit, flux_qubit] = ar


@dataclass
class QubitCrosstalkResults(QubitFluxResults):
    """
    Qubit Crosstalk outputs.
    """

    qubit_frequency_bias_point: dict[QubitId, float] = field(default_factory=dict)
    """Expected qubit frequency at bias point."""
    crosstalk_matrix: dict[QubitId, dict[QubitId, float]] = field(default_factory=dict)
    """Crosstalk matrix element."""
    fitted_parameters: dict[tuple[QubitId, QubitId], dict] = field(default_factory=dict)
    """Fitted parameters for each couple target-flux qubit."""

    def __contains__(self, key: QubitId):
        """Checking if qubit is in crosstalk_matrix attribute."""
        return key in self.crosstalk_matrix


def _acquisition(
    params: QubitCrosstalkParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> QubitCrosstalkData:
    """Data acquisition for Crosstalk Experiment."""

    assert set(targets).isdisjoint(set(params.flux_qubits)), (
        "Flux qubits must be different from targets."
    )

    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    offset = {}
    charging_energy = {}
    matrix_element = {}
    maximum_frequency = {}
    freq_sweepers = []
    offset_sweepers = []

    for qubit in targets:
        natives = platform.natives.single_qubit[qubit]
        charging_energy[qubit] = platform.calibration.single_qubits[
            qubit
        ].qubit.charging_energy
        qd_channel, qd_pulse = natives.RX()[0]
        ro_channel, ro_pulse = natives.MZ()[0]

        qd_pulse = replace(qd_pulse, duration=params.drive_duration)
        if params.drive_amplitude is not None:
            qd_pulse = replace(qd_pulse, amplitude=params.drive_amplitude)

        qd_pulses[qubit] = qd_pulse
        ro_pulses[qubit] = ro_pulse

        # store calibration parameters
        maximum_frequency[qubit] = platform.calibration.single_qubits[
            qubit
        ].qubit.maximum_frequency
        matrix_element[qubit] = platform.calibration.get_crosstalk_element(qubit, qubit)
        charging_energy[qubit] = platform.calibration.single_qubits[
            qubit
        ].qubit.charging_energy
        offset[qubit] = (
            -platform.calibration.single_qubits[qubit].qubit.sweetspot
            * matrix_element[qubit]
        )

        sequence.append((qd_channel, qd_pulse))
        sequence.append((ro_channel, Delay(duration=qd_pulse.duration)))
        sequence.append((ro_channel, ro_pulse))

        freq_sweepers.append(
            Sweeper(
                parameter=Parameter.frequency,
                values=platform.config(qd_channel).frequency + params.frequency_range,
                channels=[qd_channel],
            )
        )

    for q in params.flux_qubits:
        flux_channel = platform.qubits[q].flux
        offset0 = platform.config(flux_channel).offset
        offset_sweepers.append(
            Sweeper(
                parameter=Parameter.offset,
                values=offset0 + params.bias_range,
                channels=[flux_channel],
            )
        )

    data = QubitCrosstalkData(
        resonator_type=platform.resonator_type,
        matrix_element=matrix_element,
        offset=offset,
        qubit_frequency=maximum_frequency,
        charging_energy=charging_energy,
        bias_point=params.bias_point,
    )

    options = dict(
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    updates = []
    for qubit in targets:
        if qubit in params.bias_point:
            channel = platform.qubits[qubit].flux
            updates.append({channel: {"offset": params.bias_point[qubit]}})

    updates += [
        {platform.qubits[q].probe: {"frequency": readout_frequency(q, platform)}}
        for q in targets
    ]

    for flux_qubit, offset_sweeper in zip(params.flux_qubits, offset_sweepers):
        results = platform.execute(
            [sequence], [[offset_sweeper], freq_sweepers], **options, updates=updates
        )

        # retrieve the results for every qubit
        for i, qubit in enumerate(targets):
            result = results[ro_pulses[qubit].id]
            data.register_qubit(
                qubit,
                flux_qubit,
                signal=magnitude(result),
                freq=freq_sweepers[i].values,
                bias=offset_sweeper.values,
            )
    return data


def _fit(data: QubitCrosstalkData) -> QubitCrosstalkResults:
    crosstalk_matrix = {qubit: {} for qubit in data.qubit_frequency}
    fitted_parameters = {}
    qubit_frequency_bias_point = {}

    for target_flux_qubit, qubit_data in data.data.items():
        frequencies, biases = extract_feature(
            qubit_data.freq,
            qubit_data.bias,
            qubit_data.signal,
            "max" if data.resonator_type == "2D" else "min",
        )
        target_qubit, flux_qubit = target_flux_qubit

        qubit_frequency_bias_point[target_qubit] = (
            utils.transmon_frequency(
                xi=data.bias_point[target_qubit],
                xj=0,
                d=0,
                w_max=data.qubit_frequency[target_qubit] * HZ_TO_GHZ,
                offset=data.offset[target_qubit],
                normalization=data.matrix_element[target_qubit],
                charging_energy=data.charging_energy[target_qubit] * HZ_TO_GHZ,
                crosstalk_element=1,
            )
            * GHZ_TO_HZ
        )

        def fit_function(x, crosstalk_element, offset):
            return utils.transmon_frequency(
                xi=data.bias_point[target_qubit],
                xj=x,
                d=0,
                w_max=data.qubit_frequency[target_qubit] * HZ_TO_GHZ,
                offset=offset,
                normalization=data.matrix_element[target_qubit],
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
                xi=data.bias_point[target_qubit],
                d=0,
                w_max=data.qubit_frequency[target_qubit] * HZ_TO_GHZ,
                offset=popt[1],
                normalization=data.matrix_element[target_qubit],
                charging_energy=data.charging_energy[target_qubit] * HZ_TO_GHZ,
                crosstalk_element=float(popt[0]),
            )
            crosstalk_matrix[target_qubit][flux_qubit] = (
                popt[0] * data.matrix_element[target_qubit]
            )

        except (RuntimeError, ValueError) as e:  # pragma: no cover
            log.error(f"Error in qubit_crosstalk protocol fit: {e} ")

    return QubitCrosstalkResults(
        qubit_frequency_bias_point=qubit_frequency_bias_point,
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
            "Qubit Frequency at Bias point [Hz]",
        ]
        values = [
            np.round(fit.qubit_frequency_bias_point[target], 4),
        ]
        for flux_qubit in fit.crosstalk_matrix[target]:
            if flux_qubit != target:
                labels.append(f"Crosstalk with qubit {flux_qubit} [V^-1]")
            else:
                labels.append("Flux dependence [V^-1]")
            values.append(np.round(fit.crosstalk_matrix[target][flux_qubit], 4))
        fitting_report = table_html(
            table_dict(
                target,
                labels,
                values,
            )
        )
    return figures, fitting_report


def _update(
    results: QubitCrosstalkResults, platform: CalibrationPlatform, qubit: QubitId
):
    """Update crosstalk matrix."""

    for flux_qubit, element in results.crosstalk_matrix[qubit].items():
        update.crosstalk_matrix(element, platform, qubit, flux_qubit)


qubit_crosstalk = Routine(_acquisition, _fit, _plot, _update)
"""Qubit crosstalk Routine object"""
