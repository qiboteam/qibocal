"""SWAP experiment for two qubit gates, chevron plot."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper

from qibocal.auto.operation import QubitPairId, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.result import magnitude

from ...utils import readout_frequency
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


ChevronSignalType = np.dtype(
    [
        ("amp", np.float64),
        ("length", np.float64),
        ("signal_high", np.float64),
        ("signal_low", np.float64),
    ]
)
"""Custom dtype for Chevron."""


@dataclass
class ChevronSignalData(ChevronData):
    """Chevron acquisition outputs."""

    data: dict[QubitPairId, npt.NDArray[ChevronSignalType]] = field(
        default_factory=dict
    )

    def register_qubit(
        self, low_qubit, high_qubit, length, amp, signal_low, signal_high
    ):
        """Store output for single qubit."""
        size = len(length) * len(amp)
        amplitude, duration = np.meshgrid(amp, length)
        ar = np.empty(size, dtype=ChevronSignalType)
        ar["length"] = duration.ravel()
        ar["amp"] = amplitude.ravel()
        ar["signal_low"] = signal_low.ravel()
        ar["signal_high"] = signal_high.ravel()
        self.data[low_qubit, high_qubit] = np.rec.array(ar)

    def low_frequency(self, pair):
        return self[pair].signal_low

    def high_frequency(self, pair):
        return self[pair].signal_high


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
        sequence, flux_pulse, parking_pulses, delays = chevron_sequence(
            platform=platform,
            ordered_pair=ordered_pair,
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

        ro_high = list(sequence.channel(platform.qubits[ordered_pair[1]].acquisition))[
            -1
        ]
        ro_low = list(sequence.channel(platform.qubits[ordered_pair[0]].acquisition))[
            -1
        ]

        data.native_amplitude[ordered_pair] = flux_pulse.amplitude

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
        data.register_qubit(
            ordered_pair[0],
            ordered_pair[1],
            sweeper_duration.values,
            sweeper_amplitude.values,
            magnitude(results[ro_low.id]),
            magnitude(results[ro_high.id]),
        )
    return data


chevron_signal = Routine(_aquisition, _fit, _plot, _update, two_qubit_gates=True)
"""Chevron routine."""
