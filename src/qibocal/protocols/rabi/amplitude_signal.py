from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Protocol, QubitId, Results
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import readout_frequency
from qibocal.result import collect
from sklearn.decomposition import PCA

from . import utils

__all__ = [
    "RabiAmpSignalType",
    "RabiAmplitudeSignalData",
    "RabiAmplitudeSignalParameters",
    "RabiAmplitudeSignalResults",
    "_fit",
    "rabi_amplitude_signal",
]


@dataclass
class RabiAmplitudeSignalParameters(Parameters):
    """RabiAmplitude runcard inputs."""

    min_amp: float
    """Minimum amplitude."""
    max_amp: float
    """Maximum amplitude."""
    step_amp: float
    """Step amplitude."""
    pulse_length: float | None = None
    """RX pulse duration [ns]."""
    rx90: bool = False
    """Calibration of native pi pulse, if true calibrates pi/2 pulse"""


@dataclass
class RabiAmplitudeSignalResults(Results):
    """RabiAmplitude outputs."""

    amplitude: dict[QubitId, float | list[float]]
    """Drive amplitude for each qubit."""
    length: dict[QubitId, float | list[float]]
    """Drive pulse duration. Same for all qubits."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitted parameters."""
    rx90: bool
    """Pi or Pi_half calibration"""


RabiAmpSignalType = np.dtype(
    # [("amp", np.float64), ("i", np.float64), ("q", np.float64)]
    [("amp", np.float64), ("signal", np.float64), ("phase", np.float64)]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class RabiAmplitudeSignalData(Data):
    """RabiAmplitudeSignal data acquisition."""

    rx90: bool
    """Pi or Pi_half calibration"""
    durations: dict[QubitId, float] = field(default_factory=dict)
    """Pulse durations provided by the user."""
    data: dict[QubitId, npt.NDArray[RabiAmpSignalType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: RabiAmplitudeSignalParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> RabiAmplitudeSignalData:
    r"""
    Data acquisition for Rabi experiment sweeping amplitude.
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse amplitude
    to find the drive pulse amplitude that creates a rotation of a desired angle.
    """

    # create a sequence of pulses for the experiment
    sequence, qd_pulses, ro_pulses, durations = utils.sequence_amplitude(
        targets, params, platform, params.rx90
    )

    sweeper = Sweeper(
        parameter=Parameter.amplitude,
        range=(params.min_amp, params.max_amp, params.step_amp),
        pulses=[qd_pulses[qubit] for qubit in targets],
    )

    data = RabiAmplitudeSignalData(durations=durations, rx90=params.rx90)

    # sweep the parameter
    results = platform.execute(
        [sequence],
        [[sweeper]],
        updates=[
            {platform.qubits[q].probe: {"frequency": readout_frequency(q, platform)}}
            for q in targets
        ],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    for qubit in targets:
        result = results[ro_pulses[qubit].id]
        data.register_qubit(
            RabiAmpSignalType,
            (qubit),
            {
                "amp": sweeper.values,
                "i": result[..., 0],
                "q": result[..., 1],
            },
        )
    return data


def _fit(data: RabiAmplitudeSignalData) -> RabiAmplitudeSignalResults:
    """Post-processing for RabiAmplitude."""
    qubits = data.qubits

    pi_pulse_amplitudes = {}
    fitted_parameters = {}

    for qubit in qubits:
        qubit_data = data[qubit]

        rabi_parameter = qubit_data.amp

        #quadratures = collect(qubit_data.i, qubit_data.q)
        i = qubit_data.signal * np.cos(qubit_data.phase)
        q = qubit_data.signal * np.sin(qubit_data.phase)
        quadratures = collect(i, q)

        # initialize PCA instance and fit it to the quadrature data
        # and rotate the data along the principal axes
        principal_axis_signal = PCA().fit_transform(quadratures)[:, 0]

        pguess = utils.rabi_initial_guess(rabi_parameter, principal_axis_signal, "amp", signal=True)
        try:
            popt, _, pi_pulse_parameter = utils.fit_amplitude_function(
                rabi_parameter,
                principal_axis_signal,
                pguess,
                signal=True,
            )
            pi_pulse_amplitudes[qubit] = pi_pulse_parameter
            fitted_parameters[qubit] = popt

        except Exception as e:
            log.warning(f"Rabi fit failed for qubit {qubit} due to {e}.")

    return RabiAmplitudeSignalResults(
        pi_pulse_amplitudes, data.durations, fitted_parameters, data.rx90
    )


def _plot(
    data: RabiAmplitudeSignalData,
    target: QubitId,
    fit: RabiAmplitudeSignalResults = None,
):
    """Plotting function for RabiAmplitude."""
    return utils.plot(data, target, fit, data.rx90)


def _update(
    results: RabiAmplitudeSignalResults, platform: CalibrationPlatform, target: QubitId
):
    update.drive_amplitude(results.amplitude[target], results.rx90, platform, target)
    update.drive_duration(results.length[target], results.rx90, platform, target)


rabi_amplitude_signal = Protocol(_acquisition, _fit, _plot, _update)
"""RabiAmplitude Protocol object."""
