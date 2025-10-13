"""SWAP experiment for two qubit gates, chevron plot."""

from dataclasses import dataclass

import numpy as np
from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper

from qibocal.auto.operation import QubitPairId, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.result import magnitude

from ...utils import HZ_TO_GHZ, readout_frequency
from ..utils import order_pair
from .chevron import (
    ChevronData,
    ChevronParameters,
    ChevronResults,
    _fit,
    _plot,
    _update,
)
from .utils import chevron_sequence, z_normalization

__all__ = ["chevron_signal"]


@dataclass
class ChevronSignalParameters(ChevronParameters):
    """ChevronSignal runcard inputs."""


@dataclass
class ChevronSignalResults(ChevronResults):
    """ChevronSignal outputs when fitting will be done."""


@dataclass
class ChevronSignalData(ChevronData):
    """Chevron acquisition outputs."""


def _aquisition(
    params: ChevronSignalParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> ChevronSignalData:
    r"""
    Perform an iSWAP/CZ experiment between pairs of qubits by changing its frequency.

    Args:
        params: Experiment parameters.
        platform: CalibrationPlatform to use.
        targets (list): List of pairs to use sequentially.

    Returns:
        ChevronData: Acquisition data.
    """

    data = ChevronSignalData(
        native=params.native,
        _sorted_pairs=[order_pair(pair, platform) for pair in targets],
        amplitude=np.arange(
            params.amplitude_min, params.amplitude_max, params.amplitude_step
        ).tolist(),
        duration=np.arange(
            params.duration_min, params.duration_max, params.duration_step
        ).tolist(),
    )

    data.flux_coefficient = {
        pair: platform.calibration.single_qubits[pair[1]].qubit.flux_coefficients[0]
        for pair in data.sorted_pairs
    }
    data.detuning = {
        pair: (
            platform.calibration.single_qubits[pair[1]].qubit.frequency_01
            - platform.calibration.single_qubits[pair[0]].qubit.frequency_01
        )
        * HZ_TO_GHZ
        for pair in data.sorted_pairs
    }
    for pair in data.sorted_pairs:
        sequence, flux_pulse, parking_pulses, delays = chevron_sequence(
            platform=platform,
            ordered_pair=pair,
            duration_max=params.duration_max,
            parking=params.parking,
            dt=params.dt,
            native=params.native,
        )
        sweeper_amplitude = Sweeper(
            parameter=Parameter.amplitude,
            range=(params.amplitude_min, params.amplitude_max, params.amplitude_step),
            pulses=[flux_pulse],
        )
        sweeper_duration = Sweeper(
            parameter=Parameter.duration,
            range=(params.duration_min, params.duration_max, params.duration_step),
            pulses=[flux_pulse] + delays + parking_pulses,
        )

        ro_high = list(sequence.channel(platform.qubits[pair[1]].acquisition))[-1]
        ro_low = list(sequence.channel(platform.qubits[pair[0]].acquisition))[-1]

        results = platform.execute(
            [sequence],
            [[sweeper_duration], [sweeper_amplitude]],
            updates=[
                {
                    platform.qubits[q].probe: {
                        "frequency": readout_frequency(q, platform)
                    }
                }
                for q in pair
            ],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        )

        data.data[pair[0], pair[1]] = magnitude(
            np.stack(
                [
                    z_normalization(results[ro_low.id]),
                    1 - z_normalization(results[ro_high.id])
                    if params.native == "CZ"
                    else z_normalization(results[ro_high.id]),
                ]
            )
        )

    return data


chevron_signal = Routine(_aquisition, _fit, _plot, _update, two_qubit_gates=True)
"""Chevron routine."""
