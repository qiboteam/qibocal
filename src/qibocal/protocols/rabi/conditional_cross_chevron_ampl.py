"""Rabi experiment that sweeps amplitude and frequency when toggling a spectator qubit.
In this case the Rabi pulse on the target qubit is applied on the crontrol qubit line (cross)."""

import numpy as np
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Parameter,
    PulseSequence,
    Sweeper,
)

from qibocal.auto.operation import QubitId, QubitPairId, Routine
from qibocal.calibration import CalibrationPlatform

from ..utils import readout_frequency
from .conditional_chevron_ampl import (
    ConditionalRabiChevronAmplData,
    ConditionalRabiChevronAmplParameters,
    _fit,
    _plot,
    _update,
)

__all__ = ["conditional_crossrabi_chevron_ampl"]


def _acquisition(
    params: ConditionalRabiChevronAmplParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> ConditionalRabiChevronAmplData:
    """Data acquisition for Rabi experiment sweeping length."""

    if any(isinstance(x, (str, int)) for x in targets):
        raise ValueError("At least one target is not a QubitPairId type.")

    target_qubits_list, spectator_qubits_list = map(list, zip(*targets))
    if any(s in target_qubits_list for s in spectator_qubits_list):
        raise ValueError("One or multiple qubits are set as both spectator and target.")

    if len(spectator_qubits_list) != np.unique(spectator_qubits_list).size:
        raise ValueError("One or multiple spectator qubits are repeated.")

    if any([t not in platform.qubits[s].drive_extra for t, s in targets]):
        raise ValueError(
            "One or multiple target qubits are not in the cross-drive list of the spectator."
        )

    complete_sequence = PulseSequence()

    spectators_drive_dict: dict[QubitId, PulseSequence] = {}
    spectators_ro_dict: dict[QubitId, PulseSequence] = {}
    cross_rabi_dict: dict[QubitPairId, PulseSequence] = {}
    target_ro_dict: dict[QubitId, PulseSequence] = {}
    for t, s in targets:
        if s not in spectators_drive_dict and s not in spectators_ro_dict:
            spectator_natives = platform.natives.single_qubit[s]
            spectators_drive_dict[s] = spectator_natives.RX()[0]
            spectators_ro_dict[s] = spectator_natives.MZ()[0]

        if (t, s) not in cross_rabi_dict:
            cr_channel = platform.qubits[s].drive_extra[t]
            cross_rabi_dict[(t, s)] = PulseSequence(
                (cr_channel, spectators_drive_dict[s][1])
            )

        if t not in target_ro_dict:
            target_natives = platform.natives.single_qubit[t]
            target_ro_dict[t] = target_natives.MZ()[0]

    if params.activate_spectators:
        complete_sequence += PulseSequence(list(spectators_drive_dict.values()))

    # adding cross resonance pulses
    complete_sequence |= PulseSequence(list(cross_rabi_dict.values()))

    # adding readout for both spectators and targets
    complete_sequence |= PulseSequence(
        list(spectators_ro_dict.values()) + list(target_ro_dict.values())
    )

    ampl_sweeper = Sweeper(
        parameter=Parameter.amplitude,
        values=params.amplitude_range,
        pulses=[cross_rabi_dict[(t, s)][1] for t, s in targets],
    )

    freq_sweepers: dict[QubitId, Sweeper] = {}
    for t in target_qubits_list:
        target_drive_ch = platform.qubits[t].drive
        freq_sweepers[t] = Sweeper(
            parameter=Parameter.frequency,
            values=platform.config(target_drive_ch).frequency + params.frequency_range,
            channels=[target_drive_ch],
        )

    data = ConditionalRabiChevronAmplData(
        activate_spectators=params.activate_spectators
    )

    results = platform.execute(
        [complete_sequence],
        [list(freq_sweepers.values()), [ampl_sweeper]],
        updates=[
            {platform.qubits[q].probe: {"frequency": readout_frequency(q, platform)}}
            for q in target_qubits_list + spectator_qubits_list
        ],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    for pair in targets:
        target_result = results[target_ro_dict[pair[0]][1].id]
        spect_result = results[spectators_ro_dict[pair[1]][1].id]

        data.register_qubit(
            pair=pair,
            freq=freq_sweepers[pair[0]].values,
            ampl=ampl_sweeper.values,
            p_targ=target_result,
            p_spect=spect_result,
        )
    return data


conditional_crossrabi_chevron_ampl = Routine(_acquisition, _fit, _plot, _update)
"""Cross Rabi amplitude with frequency tuning and spectator effect."""
