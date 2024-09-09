from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.config import log
from qibocal.protocols.ramsey.ramsey_signal import RamseySignalParameters, RamseySignalResults, RamseySignalData
from qibocal.protocols.ramsey.ramsey_signal import _plot as qq_plot
from qibocal.protocols.ramsey.ramsey_signal import _fit as qq_fit

from ..utils import GHZ_TO_HZ, table_dict, table_html
from .utils import fitting, ramsey_fit, ramsey_sequence


RamseySignalType = np.dtype([("wait", np.float64), ("signal", np.float64)])
"""Custom dtype for coherence routines."""


def _acquisition(
    params: RamseySignalParameters,
    platform: Platform,
    targets: list[QubitId],
) -> RamseySignalData:
    """Data acquisition for Ramsey Experiment (detuned)."""
    # create a sequence of pulses for the experiment
    # RX90 - t - RX90 - MZ
    # define the parameter to sweep and its range:

    waits = np.arange(
        # wait time between RX90 pulses
        params.delay_between_pulses_start,
        params.delay_between_pulses_end,
        params.delay_between_pulses_step,
    )

    options = ExecutionParameters(
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    data = RamseySignalData(
        detuning=params.detuning,
        qubit_freqs={
            qubit: platform.qubits[qubit].native_gates.RX.frequency for qubit in targets
        },
    )

    if not params.unrolling:
        sequence = PulseSequence()

        for wait in waits:
            for qubit in targets:
                sequence += ramsey_sequence(
                    platform=platform, qubit=qubit, wait = wait, detuning=params.detuning
                )

            results = platform.execute_pulse_sequence(
            sequence,
            ExecutionParameters(
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
                averaging_mode=AveragingMode.CYCLIC,
            ),
        )
            pass

            
            sweeper = Sweeper(
                Parameter.start,
                waits,
                [
                    sequence.get_qubit_pulses(qubit).qd_pulses[-1] for qubit in targets
                ],  # TODO: check if it is correct
                type=SweeperType.ABSOLUTE,
            )

        # execute the sweep
        results = platform.sweep(
            sequence,
            options,
            sweeper,
        )

        for qubit in targets:
            result = results[sequence.get_qubit_pulses(qubit).ro_pulses[0].serial]
            # The probability errors are the standard errors of the binomial distribution
            data.register_qubit(
                RamseySignalType,
                (qubit),
                dict(
                    wait=waits,
                    signal=result.magnitude,
                ),
            )

    else:
        sequences, all_ro_pulses = [], []
        for wait in waits:
            sequence = PulseSequence()
            for qubit in targets:
                sequence += ramsey_sequence(
                    platform=platform, qubit=qubit, wait=wait, detuning=params.detuning
                )

            sequences.append(sequence)
            all_ro_pulses.append(sequence.ro_pulses)

        results = platform.execute_pulse_sequences(sequences, options)

        # We dont need ig as everty serial is different
        for ig, (wait, ro_pulses) in enumerate(zip(waits, all_ro_pulses)):
            for qubit in targets:
                serial = ro_pulses[qubit].serial
                if params.unrolling:
                    result = results[serial][0]
                else:
                    result = results[ig][serial]
                data.register_qubit(
                    RamseySignalType,
                    (qubit),
                    dict(
                        wait=np.array([wait]),
                        signal=np.array([result.magnitude]),
                    ),
                )

    return data


def _fit(data: RamseySignalData) -> RamseySignalResults:
    r"""Fitting routine for Ramsey experiment. The used model is
    .. math::

        y = p_0 + p_1 sin \Big(p_2 x + p_3 \Big) e^{-x p_4}.
    """
    return qq_fit(data)


def _plot(data: RamseySignalData, target: QubitId, fit: RamseySignalResults = None):
    """Plotting function for Ramsey Sequenes Experiment."""
    return qq_plot(data, target, fit)


def _update(results: RamseySignalResults, platform: Platform, target: QubitId):
    update.drive_frequency(results.frequency[target][0], platform, target)


ramsey_signal_sequences = Routine(_acquisition, _fit, _plot, _update)
"""Ramsey Routine object to use sequencer."""
