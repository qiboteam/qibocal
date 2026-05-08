from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go

from qibocal.auto.operation import QubitId, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.result import collect, magnitude, unpack

from .acquisition import (
    RamseyData,
    RamseyParameters,
    RamseyResults,
    execute_experiment,
    ramsey_sequence,
)
from .processing import (
    fitting,
    process_fit,
    ramsey_update,
    signal_plot,
)

__all__ = ["ramsey_signal"]


RamseySignalType = np.dtype(
    [("wait", np.float64), ("i", np.float64), ("q", np.float64)]
)
"""Custom dtype for coherence routines."""


@dataclass
class RamseySignalData(RamseyData):
    """Ramsey acquisition outputs."""

    data: dict[QubitId, npt.NDArray[RamseySignalType]] = field(default_factory=dict)
    """Raw data acquired."""

    def qubit_signal(self, qubit: QubitId) -> npt.NDArray[np.float64]:
        """
        Return the signal magnitude for a given qubit.
        """
        return magnitude(collect(self.data[qubit].i, self.data[qubit].q))


def _acquisition(
    params: RamseyParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> RamseySignalData:
    """Data acquisition for Ramsey Experiment (detuned)."""
    # create a sequence of pulses for the experiment
    # RX90 - t - RX90 - MZ
    # define the parameter to sweep and its range:

    data = RamseySignalData(
        detuning=params.detuning,
        qubit_freqs={
            qubit: platform.config(platform.qubits[qubit].drive).frequency
            for qubit in targets
        },
    )

    sequence, delays = ramsey_sequence(platform, targets)

    results = execute_experiment(
        sequence=sequence,
        delays=delays,
        platform=platform,
        targets=targets,
        params=params,
        return_probs=False,
    )

    for qubit in targets:
        ro_pulse = list(sequence.channel(platform.qubits[qubit].acquisition))[-1]
        i, q = unpack(results[ro_pulse.id])
        data.register_qubit(
            RamseySignalType,
            (qubit),
            dict(
                wait=np.arange(*params.delay_range),
                i=i,
                q=q,
            ),
        )

    return data


def _fit(data: RamseySignalData) -> RamseyResults:
    r"""Fitting routine for Ramsey experiment. The used model is
    .. math::

        y = p_0 + p_1 sin \Big(p_2 x + p_3 \Big) e^{-x p_4}.
    """
    qubits = data.qubits
    waits = data.waits
    popts: dict[QubitId, list[float]] = {}
    freq_measure: dict[QubitId, list[float]] = {}
    t2_measure: dict[QubitId, list[float]] = {}
    delta_phys_measure: dict[QubitId, list[float]] = {}
    delta_fitting_measure: dict[QubitId, list[float]] = {}
    for qubit in qubits:
        qubit_freq = data.qubit_freqs[qubit]
        signal = data.qubit_signal(qubit)
        try:
            popt, perr = fitting(waits, signal)
            (
                freq_measure[qubit],
                t2_measure[qubit],
                delta_phys_measure[qubit],
                delta_fitting_measure[qubit],
                popts[qubit],
            ) = process_fit(popt, perr, qubit_freq, data.detuning)
        except Exception as e:
            log.warning(f"Ramsey fitting failed for qubit {qubit} due to {e}.")

    return RamseyResults(
        detuning=data.detuning,
        frequency=freq_measure,
        t2=t2_measure,
        delta_phys=delta_phys_measure,
        delta_fitting=delta_fitting_measure,
        fitted_parameters=popts,
    )


def _plot(
    data: RamseySignalData, target: QubitId, fit: RamseyResults | None = None
) -> tuple[list[go.Figure], str]:
    """Plotting function for Ramsey Experiment."""

    return signal_plot(
        waits=data.waits,
        signal=data.qubit_signal(target),
        target=target,
        fit=fit,
        yaxis_title="Signal [a.u.]",
    )


ramsey_signal = Routine(_acquisition, _fit, _plot, ramsey_update)
"""Ramsey Routine object.

The protocol consists in applying the following pulse sequence:
RX90 - wait - RX90 - MZ
for different waiting times `wait`.
The range of waiting times is defined through the attributes
`delay_between_pulses_*` available in `RamseyParameters`. The final range
will be constructed using `np.arange`.
It is possible to detune the drive frequency using the parameter `detuning` in
RamseyParameters which will increment the drive frequency accordingly.
Currently when `detuning==0` it will be performed a sweep over the waiting values
if `detuning` is not zero, all sequences with different waiting value will be
executed sequentially.
The following protocol will display on the y-axis the signal amplitude.
"""
