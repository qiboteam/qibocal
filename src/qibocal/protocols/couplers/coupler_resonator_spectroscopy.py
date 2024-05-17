import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitPairId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Routine

from ..flux_dependence.utils import flux_dependence_plot
from ..two_qubit_interaction.utils import order_pair
from .utils import (
    CouplerSpectroscopyData,
    CouplerSpectroscopyParameters,
    CouplerSpectroscopyResults,
)


def _acquisition(
    params: CouplerSpectroscopyParameters,
    platform: Platform,
    targets: list[QubitPairId],
) -> CouplerSpectroscopyData:
    """
    Data acquisition for CouplerResonator spectroscopy.

    This consist on a frequency sweep on the readout frequency while we change the flux coupler pulse amplitude of
    the coupler pulse. We expect to enable the coupler during the amplitude sweep and detect an avoided crossing
    that will be followed by the frequency sweep. No need to have the qubits at resonance. This should be run after
    resonator_spectroscopy to detect couplers and adjust the coupler sweetspot if needed and get some information
    on the flux coupler pulse amplitude requiered to enable 2q interactions.

    """

    # TODO: Do we  want to measure both qubits on the pair ?

    # create a sequence of pulses for the experiment:
    # Coupler pulse while MZ

    if params.measured_qubits is None:
        params.measured_qubits = [order_pair(pair, platform)[0] for pair in targets]

    sequence = PulseSequence()
    ro_pulses = {}
    offset = {}
    couplers = []
    for i, pair in enumerate(targets):
        ordered_pair = order_pair(pair, platform)
        measured_qubit = params.measured_qubits[i]

        qubit = platform.qubits[measured_qubit].name
        offset[qubit] = platform.pairs[tuple(sorted(ordered_pair))].coupler.sweetspot
        coupler = platform.pairs[tuple(sorted(ordered_pair))].coupler.name
        couplers.append(coupler)
        # TODO: May measure both qubits on the pair
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        if params.amplitude is not None:
            ro_pulses[qubit].amplitude = params.amplitude

        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(
        -params.freq_width / 2, params.freq_width / 2, params.freq_step
    )

    sweeper_freq = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[ro_pulses[qubit] for qubit in params.measured_qubits],
        type=SweeperType.OFFSET,
    )

    delta_bias_range = np.arange(
        -params.bias_width / 2, params.bias_width / 2, params.bias_step
    )
    sweepers = [
        Sweeper(
            Parameter.bias,
            delta_bias_range,
            qubits=couplers,
            type=SweeperType.OFFSET,
        )
    ]

    data = CouplerSpectroscopyData(
        resonator_type=platform.resonator_type,
        offset=offset,
    )

    for bias_sweeper in sweepers:
        results = platform.sweep(
            sequence,
            ExecutionParameters(
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
                averaging_mode=AveragingMode.CYCLIC,
            ),
            bias_sweeper,
            sweeper_freq,
        )

    # retrieve the results for every qubit
    for i, pair in enumerate(targets):
        # TODO: May measure both qubits on the pair
        qubit = platform.qubits[params.measured_qubits[i]].name
        result = results[ro_pulses[qubit].serial]
        # store the results
        data.register_qubit(
            qubit,
            signal=result.magnitude,
            phase=result.phase,
            freq=delta_frequency_range + ro_pulses[qubit].frequency,
            bias=delta_bias_range,
        )
    return data


def _fit(data: CouplerSpectroscopyData) -> CouplerSpectroscopyResults:
    """Post-processing function for CouplerResonatorSpectroscopy."""
    qubits = data.qubits
    pulse_amp = {}
    sweetspot = {}
    fitted_parameters = {}

    # TODO: Implement fit
    """It should get two things:
    Coupler sweetspot: the value that makes both features centered and symmetric
    Pulse_amp: That turn on the feature taking into account the shift introduced by the coupler sweetspot
    """

    return CouplerSpectroscopyResults(
        pulse_amp=pulse_amp,
        sweetspot=sweetspot,
        fitted_parameters=fitted_parameters,
    )


def _plot(
    data: CouplerSpectroscopyData,
    target: QubitPairId,
    fit: CouplerSpectroscopyResults,
):
    """
    We may want to measure both qubits on the pair,
    that will require a different plotting that takes both.
    """
    qubit_pair = target  # TODO: Patch for 2q gate routines

    for qubit in qubit_pair:
        if qubit in data.data.keys():
            fig = flux_dependence_plot(data, fit, qubit)[0]

            fig.layout.annotations[0].update(
                text="Signal [a.u.] Qubit" + str(qubit),
            )
            fig.layout.annotations[1].update(
                text="Phase [rad] Qubit" + str(qubit),
            )

    return [fig], ""


def _update(
    results: CouplerSpectroscopyResults, platform: Platform, target: QubitPairId
):
    pass


coupler_resonator_spectroscopy = Routine(_acquisition, _fit, _plot, _update)
"""CouplerResonatorSpectroscopy Routine object."""
