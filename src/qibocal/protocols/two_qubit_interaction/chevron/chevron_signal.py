"""SWAP experiment for two qubit gates, chevron plot."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.qubits import QubitPairId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Routine

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
    platform: Platform,
    targets: list[QubitPairId],
) -> ChevronSignalData:
    r"""
    Perform an iSWAP/CZ experiment between pairs of qubits by changing its frequency.

    Args:
        params: Experiment parameters.
        platform: Platform to use.
        targets (list): List of pairs to use sequentially.

    Returns:
        ChevronData: Acquisition data.
    """

    # create a DataUnits object to store the results,
    data = ChevronSignalData(native=params.native)
    for pair in targets:
        # order the qubits so that the low frequency one is the first
        ordered_pair = order_pair(pair, platform)
        sequence = chevron_sequence(
            platform=platform,
            pair=pair,
            duration_max=params.duration_max,
            parking=params.parking,
            dt=params.dt,
            native=params.native,
        )

        data.native_amplitude[ordered_pair] = (
            sequence.get_qubit_pulses(ordered_pair[1]).qf_pulses[0].amplitude
        )
        data.sweetspot[ordered_pair] = platform.qubits[ordered_pair[1]].sweetspot

        sweeper_amplitude = Sweeper(
            Parameter.amplitude,
            params.amplitude_range,
            pulses=[sequence.get_qubit_pulses(ordered_pair[1]).qf_pulses[0]],
            type=SweeperType.FACTOR,
        )
        sweeper_duration = Sweeper(
            Parameter.duration,
            params.duration_range,
            pulses=[sequence.get_qubit_pulses(ordered_pair[1]).qf_pulses[0]],
            type=SweeperType.ABSOLUTE,
        )
        results = platform.sweep(
            sequence,
            ExecutionParameters(
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
                averaging_mode=AveragingMode.CYCLIC,
            ),
            sweeper_duration,
            sweeper_amplitude,
        )
        data.register_qubit(
            ordered_pair[0],
            ordered_pair[1],
            params.duration_range,
            params.amplitude_range * data.native_amplitude[ordered_pair],
            results[ordered_pair[0]].magnitude,
            results[ordered_pair[1]].magnitude,
        )
    return data


chevron_signal = Routine(_aquisition, _fit, _plot, _update, two_qubit_gates=True)
"""Chevron routine."""
