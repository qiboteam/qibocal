import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal import update
from qibocal.auto.operation import Qubits, Routine

from ..flux_dependence.utils import flux_dependence_plot
from ..two_qubit_interaction.utils import order_pair
from .utils import (
    CouplerSpectroscopyData,
    CouplerSpectroscopyParameters,
    CouplerSpectroscopyResults,
)


def _acquisition(
    params: CouplerSpectroscopyParameters, platform: Platform, qubits: Qubits
) -> CouplerSpectroscopyData:
    """Data acquisition for CouplerResonator spectroscopy."""
    # create a sequence of pulses for the experiment:
    # Coupler pulse - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}

    for pair in qubits:
        # TODO: DO general
        qubit = platform.qubits[params.measured_qubit].name
        # TODO: Qubit pair patch
        ordered_pair = order_pair(pair, platform.qubits)
        coupler = platform.pairs[tuple(sorted(ordered_pair))].coupler

        # TODO: Does it need time or can it start at 0 ???
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=1000)
        if params.amplitude is not None:
            ro_pulses[qubit].amplitude = params.amplitude

        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )

    # TODO: fix loop
    sweeper_freq = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[ro_pulses[qubit] for qubit in [params.measured_qubit]],
        type=SweeperType.OFFSET,
    )

    # define the parameter to sweep and its range:
    delta_bias_range = np.arange(
        -params.bias_width / 2, params.bias_width / 2, params.bias_step
    )

    # TODO: fix loop
    sweeper_bias = Sweeper(
        Parameter.bias,
        delta_bias_range,
        couplers=[coupler],
        type=SweeperType.ABSOLUTE,
    )

    data = CouplerSpectroscopyData(
        resonator_type=platform.resonator_type,
    )

    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        sweeper_bias,
        sweeper_freq,
    )

    # TODO: fix loop
    # retrieve the results for every qubit
    for pair in qubits:
        # TODO: DO general
        qubit = platform.qubits[params.measured_qubit].name
        # average msr, phase, i and q over the number of shots defined in the runcard
        result = results[ro_pulses[qubit].serial]
        # store the results
        data.register_qubit(
            qubit,
            msr=result.magnitude,
            phase=result.phase,
            freq=delta_frequency_range + ro_pulses[qubit].frequency,
            bias=delta_bias_range,
        )
    return data


def _fit(data: CouplerSpectroscopyData) -> CouplerSpectroscopyResults:
    """Post-processing function for CouplerResonatorSpectroscopy."""
    qubits = data.qubits
    sweetspot = {}
    fitted_parameters = {}

    for qubit in qubits:
        # TODO: Fix fit
        # freq, fitted_params = lorentzian_fit(
        #     data[qubit], resonator_type=data.resonator_type, fit="resonator"
        # )

        sweetspot[qubit] = 0
        fitted_parameters[qubit] = {}

    return CouplerSpectroscopyResults(
        sweetspot=sweetspot,
        fitted_parameters=fitted_parameters,
    )


def _plot(
    data: CouplerSpectroscopyData,
    qubit,
    fit: CouplerSpectroscopyResults,
):
    """Plotting function for CouplerResonatorSpectroscopy."""
    # TODO: fix loop
    for q in qubit:
        if q != 2:
            return flux_dependence_plot(data, fit, q)


def _update(results: CouplerSpectroscopyResults, platform: Platform, qubit: QubitId):
    # TODO: Need Couplers in the update
    if 1 == 0:
        update.coupler_sweetspot(results.sweetspot[qubit], platform, qubit)


coupler_resonator_spectroscopy = Routine(_acquisition, _fit, _plot, _update)
"""CouplerResonatorSpectroscopy Routine object."""
