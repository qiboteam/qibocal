from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    Platform,
    PulseSequence,
    Sweeper,
)
from scipy.optimize import curve_fit

from qibocal.auto.operation import Data, QubitId, Results, Routine
from qibocal.config import log
from qibocal.result import magnitude, phase
from qibocal.update import replace

from ... import update
from ..utils import GHZ_TO_HZ, HZ_TO_GHZ, extract_feature, table_dict, table_html
from . import utils
from .resonator_flux_dependence import ResonatorFluxParameters


@dataclass
class QubitFluxParameters(ResonatorFluxParameters):
    """QubitFlux runcard inputs."""

    drive_amplitude: Optional[float] = None
    """Drive amplitude (optional). If defined, same amplitude will be used in all qubits.
    Otherwise the default amplitude defined on the platform runcard will be used"""
    drive_duration: int = 2000
    """Duration of the drive pulse."""


@dataclass
class QubitFluxResults(Results):
    """QubitFlux outputs."""

    sweetspot: dict[QubitId, float] = field(default_factory=dict)
    """Sweetspot for each qubit."""
    frequency: dict[QubitId, float] = field(default_factory=dict)
    """Drive frequency for each qubit."""
    fitted_parameters: dict[QubitId, dict[str, float]] = field(default_factory=dict)
    """Raw fitting output."""
    matrix_element: dict[QubitId, float] = field(default_factory=dict)
    """V_ii coefficient."""


QubitFluxType = np.dtype(
    [
        ("freq", np.float64),
        ("bias", np.float64),
        ("signal", np.float64),
        ("phase", np.float64),
    ]
)
"""Custom dtype for resonator flux dependence."""


@dataclass
class QubitFluxData(Data):
    """QubitFlux acquisition outputs."""

    resonator_type: str
    """Resonator type."""
    charging_energy: dict[QubitId, float] = field(default_factory=dict)
    """Qubit charging energy."""
    qubit_frequency: dict[QubitId, float] = field(default_factory=dict)
    """Qubit charging energy."""
    data: dict[QubitId, npt.NDArray[QubitFluxType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, freq, bias, signal, phase):
        """Store output for single qubit."""
        self.data[qubit] = utils.create_data_array(
            freq, bias, signal, phase, dtype=QubitFluxType
        )


def _acquisition(
    params: QubitFluxParameters,
    platform: Platform,
    targets: list[QubitId],
) -> QubitFluxData:
    """Data acquisition for QubitFlux Experiment."""

    delta_frequency_range = np.arange(
        -params.freq_width / 2, params.freq_width / 2, params.freq_step
    )
    delta_offset_range = np.arange(
        -params.bias_width / 2, params.bias_width / 2, params.bias_step
    )

    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    qubit_frequency = {}
    freq_sweepers = []
    offset_sweepers = []
    for q in targets:
        natives = platform.natives.single_qubit[q]
        qd_channel, qd_pulse = natives.RX()[0]
        ro_channel, ro_pulse = natives.MZ()[0]

        qd_pulse = replace(qd_pulse, duration=params.drive_duration)
        if params.drive_amplitude is not None:
            qd_pulse = replace(qd_pulse, amplitude=params.drive_amplitude)

        qd_pulses[q] = qd_pulse
        ro_pulses[q] = ro_pulse
        qubit_frequency[q] = frequency0 = platform.config(qd_channel).frequency

        sequence.append((qd_channel, qd_pulse))
        sequence.append((ro_channel, Delay(duration=qd_pulse.duration)))
        sequence.append((ro_channel, ro_pulse))

        # define the parameters to sweep and their range:
        freq_sweepers.append(
            Sweeper(
                parameter=Parameter.frequency,
                values=frequency0 + delta_frequency_range,
                channels=[qd_channel],
            )
        )

        flux_channel = platform.qubits[q].flux
        offset0 = platform.config(flux_channel).offset
        offset_sweepers.append(
            Sweeper(
                parameter=Parameter.offset,
                values=offset0 + delta_offset_range,
                channels=[flux_channel],
            )
        )

    data = QubitFluxData(
        resonator_type=platform.resonator_type,
        charging_energy={
            qubit: platform.calibration.single_qubits[qubit].qubit.charging_energy
            for qubit in targets
        },
        qubit_frequency=qubit_frequency,
    )
    results = platform.execute(
        [sequence],
        [offset_sweepers, freq_sweepers],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

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


def _fit(data: QubitFluxData) -> QubitFluxResults:
    """
    Post-processing for QubitFlux Experiment. See arxiv:0703002
    Fit frequency as a function of current for the flux qubit spectroscopy
    data (QubitFluxData): data object with information on the feature response at each current point.
    """

    qubits = data.qubits
    frequency = {}
    sweetspot = {}
    matrix_element = {}
    fitted_parameters = {}

    for qubit in qubits:
        qubit_data = data[qubit]
        biases = qubit_data.bias
        frequencies = qubit_data.freq
        signal = qubit_data.signal

        frequencies, biases = extract_feature(
            frequencies, biases, signal, "max" if data.resonator_type == "2D" else "min"
        )

        def fit_function(x, w_max, normalization, offset):
            return utils.transmon_frequency(
                xi=x,
                w_max=w_max,
                xj=0,
                d=0,
                normalization=normalization,
                offset=offset,
                crosstalk_element=1,
                charging_energy=data.charging_energy[qubit] * HZ_TO_GHZ,
            )

        try:
            popt = curve_fit(
                fit_function,
                biases,
                frequencies * HZ_TO_GHZ,
                bounds=utils.qubit_flux_dependence_fit_bounds(
                    data.qubit_frequency[qubit],
                ),
                maxfev=100000,
            )[0]
            fitted_parameters[qubit] = {
                "w_max": popt[0],
                "xj": 0,
                "d": 0,
                "normalization": popt[1],
                "offset": popt[2],
                "crosstalk_element": 1,
                "charging_energy": data.charging_energy[qubit] * HZ_TO_GHZ,
            }
            frequency[qubit] = popt[0] * GHZ_TO_HZ
            # solution to x*popt[1] + popt[2] = k
            # such that x is close to 0
            # to avoid errors due to periodicity
            sweetspot[qubit] = (np.round(popt[2]) - popt[2]) / popt[1]
            matrix_element[qubit] = popt[1]
        except ValueError as e:
            log.error(
                f"Error in qubit_flux protocol fit: {e} "
                "The threshold for the SNR mask is probably too high. "
                "Lowering the value of `threshold` in `extract_*_feature`"
                "should fix the problem."
            )

    return QubitFluxResults(
        frequency=frequency,
        sweetspot=sweetspot,
        matrix_element=matrix_element,
        fitted_parameters=fitted_parameters,
    )


def _plot(data: QubitFluxData, fit: QubitFluxResults, target: QubitId):
    """Plotting function for QubitFlux Experiment."""

    figures = utils.flux_dependence_plot(
        data,
        fit,
        target,
        fit_function=utils.transmon_frequency,
    )
    if fit is not None:
        fitting_report = table_html(
            table_dict(
                target,
                [
                    f"Sweetspot [V]",
                    "Qubit Frequency at Sweetspot [Hz]",
                    "Flux dependence [V]^-1",
                ],
                [
                    np.round(fit.sweetspot[target], 4),
                    np.round(fit.frequency[target], 4),
                    np.round(fit.matrix_element[target], 4),
                ],
            )
        )
        return figures, fitting_report
    return figures, ""


def _update(results: QubitFluxResults, platform: Platform, qubit: QubitId):
    # update.drive_frequency(results.frequency[qubit], platform, qubit)
    update.sweetspot(results.sweetspot[qubit], platform, qubit)
    platform.calibration.single_qubits[qubit].qubit.maximum_frequency = int(
        results.frequency[qubit]
    )
    update.crosstalk_matrix(results.matrix_element[qubit], platform, qubit, qubit)


qubit_flux = Routine(_acquisition, _fit, _plot, _update)
"""QubitFlux Routine object."""
