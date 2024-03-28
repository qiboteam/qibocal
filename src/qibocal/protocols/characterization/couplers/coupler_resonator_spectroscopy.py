from typing import Optional

import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitPairId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Routine

from ..flux_dependence import resonator_flux_dependence
from ..flux_dependence.utils import flux_dependence_plot
from ..two_qubit_interaction.utils import order_pair
from .utils import (
    CouplerSpectroscopyData,
    CouplerSpectroscopyParameters,
    CouplerSpectroscopyResults,
)


class CouplerSpectroscopyParametersResonator(CouplerSpectroscopyParameters):
    readout_delay: Optional[int] = 1000
    """Readout delay before the measurement is done to let the flux coupler pulse act"""


def _acquisition(
    params: CouplerSpectroscopyParametersResonator,
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

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    couplers = []
    for i, pair in enumerate(targets):
        qubit = platform.qubits[params.measured_qubits[i]].name
        # TODO: Qubit pair patch
        ordered_pair = order_pair(pair, platform.qubits)
        coupler = platform.pairs[tuple(sorted(ordered_pair))].coupler
        couplers.append(coupler)
        # TODO: May measure both qubits on the pair
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=params.readout_delay
        )
        if params.amplitude is not None:
            ro_pulses[qubit].amplitude = params.amplitude

        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )

    sweeper_freq = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[ro_pulses[qubit] for qubit in params.measured_qubits],
        type=SweeperType.OFFSET,
    )

    if params.flux_pulses:
        # TODO: Add delay
        (
            delta_bias_flux_range,
            sweepers,
        ) = resonator_flux_dependence.create_flux_pulse_sweepers(
            params, platform, couplers, sequence
        )
    else:
        delta_bias_flux_range = np.arange(
            -params.bias_width / 2, params.bias_width / 2, params.bias_step
        )
        sweepers = [
            Sweeper(
                Parameter.bias,
                delta_bias_flux_range,
                qubits=couplers,
                type=SweeperType.OFFSET,
            )
        ]

    data = CouplerSpectroscopyData(
        resonator_type=platform.resonator_type,
        flux_pulses=params.flux_pulses,
    )

    for bias_sweeper in sweepers:
        print(bias_sweeper)
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
            bias=delta_bias_flux_range,
        )
    return data


def _fit(data: CouplerSpectroscopyData) -> CouplerSpectroscopyResults:
    """Post-processing function for CouplerResonatorSpectroscopy."""
    qubits = data.qubits
    pulse_amp = {}
    sweetspot = {}
    fitted_parameters = {}

    for qubit in qubits:
        # TODO: Implement fit
        """It should get two things:
        Coupler sweetspot: the value that makes both features centered and symmetric
        Pulse_amp: That turn on the feature taking into account the shift introduced by the coupler sweetspot

        Issues:  Coupler sweetspot it measured in volts while pulse_amp is a pulse amplitude, this routine just sweeps pulse amplitude
        and relies on manual shifting of that sweetspot by repeated scans as current chips are already symmetric for this feature.
        Maybe another routine sweeping the bias in volts would be needed and that sweeper implement on Zurich driver.
        """
        # spot, amp, fitted_params = coupler_fit(data[qubit])

        sweetspot[qubit] = 0
        pulse_amp[qubit] = 0
        fitted_parameters[qubit] = None

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

            # if data.flux_pulses:
            #     bias_flux_unit = "a.u."
            # else:
            #     bias_flux_unit = "V"

            # if fit is not None:
            #     fitting_report = table_html(
            #         table_dict(
            #             target,
            #             [
            #                 f"Coupler activation [{bias_flux_unit}]",
            #                 f"Coupler sweetspot [{bias_flux_unit}]",
            #             ],
            #             [
            #                 np.round(fit.pulse_amp[target], 4),
            #                 np.round(fit.sweetspot[target], 4),
            #             ],
            #         )
            #     )
            # return [fig], fitting_report
    return [fig], ""


def _update(
    results: CouplerSpectroscopyResults, platform: Platform, target: QubitPairId
):
    pass


coupler_resonator_spectroscopy = Routine(_acquisition, _fit, _plot, _update)
"""CouplerResonatorSpectroscopy Routine object."""
