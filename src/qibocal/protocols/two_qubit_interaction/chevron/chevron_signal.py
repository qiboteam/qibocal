"""SWAP experiment for two qubit gates, chevron plot."""

from dataclasses import dataclass

import numpy as np
from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper
from sklearn.preprocessing import minmax_scale

from qibocal.auto.operation import Protocol, QubitPairId
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
from .utils import chevron_sequence

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

    # create a DataUnits object to store the results,
    data = ChevronSignalData(native=params.native)
    for pair in targets:
        # order the qubits so that the low frequency one is the first
        ordered_pair = order_pair(pair, platform)
        sequence, flux_pulse, coupler_pulse, parking_pulses, delays = chevron_sequence(
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

        pulses_duration_sweeper = [flux_pulse] + delays + parking_pulses
        if coupler_pulse is not None:
            pulses_duration_sweeper += [coupler_pulse]

        sweeper_duration = Sweeper(
            parameter=Parameter.duration,
            range=(params.duration_min, params.duration_max, params.duration_step),
            pulses=pulses_duration_sweeper,
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

        data.data[pair[0], pair[1]] = np.stack(
            [
                minmax_scale(magnitude(results[ro_low.id])),
                1 - minmax_scale(magnitude(results[ro_high.id]))
                if params.native == "CZ"
                else minmax_scale(magnitude(results[ro_high.id])),
            ]
        )

    return data


chevron_signal = Protocol(_aquisition, _fit, _plot, _update, two_qubit_gates=True)
"""Chevron routine."""
