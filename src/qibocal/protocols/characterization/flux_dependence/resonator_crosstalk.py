from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Qubits, Results, Routine
from qibocal.protocols.characterization.flux_dependence.resonator_flux_dependence import (
    ResFluxType,
    ResonatorFluxData,
    ResonatorFluxParameters,
)

from . import utils


@dataclass
class ResCrosstalkParameters(ResonatorFluxParameters):
    """ResonatorFlux runcard inputs."""

    flux_qubits: Optional[list[QubitId]] = None
    """IDs of the qubits that we will sweep the flux on.
    If ``None`` flux will be swept on all qubits that we are running the routine on in a multiplex fashion.
    If given flux will be swept on the given qubits in a sequential fashion (n qubits will result to n different executions).
    Multiple qubits may be measured in each execution as specified by the ``qubits`` option in the runcard.
    """


@dataclass
class ResCrosstalkResults(Results):
    """Empty fitting outputs for cross talk because fitting is not implemented in this case."""


@dataclass
class ResCrosstalkData(ResonatorFluxData):
    """QubitFlux acquisition outputs when ``flux_qubits`` are given."""

    data: dict[tuple[QubitId, QubitId], npt.NDArray[ResFluxType]] = field(
        default_factory=dict
    )
    """Raw data acquired for (qubit, qubit_flux) pairs saved in nested dictionaries."""

    def register_qubit(self, qubit, flux_qubit, freq, bias, signal, phase):
        """Store output for single qubit."""
        ar = utils.create_data_array(freq, bias, signal, phase, dtype=ResFluxType)
        if (qubit, flux_qubit) in self.data:
            self.data[qubit, flux_qubit] = np.rec.array(
                np.concatenate((self.data[qubit, flux_qubit], ar))
            )
        else:
            self.data[qubit, flux_qubit] = ar


def _acquisition(
    params: ResCrosstalkParameters, platform: Platform, qubits: Qubits
) -> ResonatorFluxData:
    """Data acquisition for ResonatorFlux experiment."""
    # create a sequence of pulses for the experiment:
    # MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    Ec = {}
    Ej = {}
    g = {}
    bare_resonator_frequency = {}
    for qubit in qubits:
        Ec[qubit] = qubits[qubit].Ec
        Ej[qubit] = qubits[qubit].Ej
        g[qubit] = qubits[qubit].g
        bare_resonator_frequency[qubit] = qubits[qubit].bare_resonator_frequency

        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulses[qubit])

    # define the parameters to sweep and their range:
    delta_frequency_range = np.arange(
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )
    freq_sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        [ro_pulses[qubit] for qubit in qubits],
        type=SweeperType.OFFSET,
    )

    delta_bias_range = np.arange(
        -params.bias_width / 2, params.bias_width / 2, params.bias_step
    )
    if params.flux_qubits is None:
        flux_qubits = list(platform.qubits.keys())

    else:
        flux_qubits = params.flux_qubits

    bias_sweepers = [
        Sweeper(
            Parameter.bias,
            delta_bias_range,
            qubits=[platform.qubits[flux_qubit]],
            type=SweeperType.OFFSET,
        )
        for flux_qubit in flux_qubits
    ]

    data = ResCrosstalkData(
        resonator_type=platform.resonator_type,
        Ec=Ec,
        Ej=Ej,
        g=g,
        bare_resonator_frequency=bare_resonator_frequency,
    )
    options = ExecutionParameters(
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    for flux_qubit, bias_sweeper in zip(flux_qubits, bias_sweepers):
        results = platform.sweep(sequence, options, bias_sweeper, freq_sweeper)
        # retrieve the results for every qubit
        for qubit in qubits:
            result = results[ro_pulses[qubit].serial]
            if flux_qubit is None:
                sweetspot = qubits[qubit].sweetspot
            else:
                sweetspot = platform.qubits[flux_qubit].sweetspot
            data.register_qubit(
                qubit,
                flux_qubit,
                signal=result.magnitude,
                phase=result.phase,
                freq=delta_frequency_range + ro_pulses[qubit].frequency,
                bias=delta_bias_range + sweetspot,
            )

    return data


def _fit(data: ResCrosstalkData) -> ResCrosstalkResults:
    return ResCrosstalkResults()


def _plot(data: ResCrosstalkData, fit: ResCrosstalkResults, qubit):
    """Plotting function for ResonatorFlux Experiment."""
    return utils.flux_crosstalk_plot(data, qubit)


resonator_crosstalk = Routine(_acquisition, _fit, _plot)
"""Resonator crosstalk Routine object"""
