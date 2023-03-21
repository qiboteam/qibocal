from dataclasses import dataclass

import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from ...auto.operation import Qubits, Results, Routine
from ...data import DataUnits
from .resonator_spectroscopy import ResonatorSpectroscopyParameters


@dataclass
class ResonatorPunchoutParameters(ResonatorSpectroscopyParameters):
    min_amp_factor: float
    max_amp_factor: float
    step_amp_factor: float


@dataclass
class ResonatorPunchoutResults(Results):
    ...


class ResonatorPunchoutData(DataUnits):
    def __init__(self):
        super().__init__(
            "data",
            {"frequency": "Hz", "amplitude": "dimensionless"},
            options=["qubit", "iteration"],
        )


def _acquisition(
    platform: AbstractPlatform, qubits: Qubits, params: ResonatorPunchoutParameters
) -> ResonatorPunchoutData:
    # reload instrument settings from runcard
    platform.reload_settings()

    # create a sequence of pulses for the experiment:
    # MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()

    ro_pulses = {}
    for qubit in qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulses[qubit])

    # define the parameters to sweep and their range:
    # resonator frequency
    delta_frequency_range = np.arange(
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )
    freq_sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        [ro_pulses[qubit] for qubit in qubits],
    )

    # amplitude
    amplitude_range = np.arange(
        params.min_amp_factor, params.max_amp_factor, params.step_amp_factor
    )
    amp_sweeper = Sweeper(
        Parameter.amplitude, amplitude_range, [ro_pulses[qubit] for qubit in qubits]
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include resonator frequency and attenuation
    data = ResonatorPunchoutData()

    # repeat the experiment as many times as defined by software_averages
    amps = np.repeat(amplitude_range, len(delta_frequency_range))
    for iteration in range(params.software_averages):
        results = platform.sweep(
            sequence,
            amp_sweeper,
            freq_sweeper,
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
        )

        # retrieve the results for every qubit
        for qubit, ro_pulse in ro_pulses.items():
            # average msr, phase, i and q over the number of shots defined in the runcard
            result = results[ro_pulse.serial]
            # store the results
            freqs = np.array(
                len(amplitude_range) * list(delta_frequency_range + ro_pulse.frequency)
            ).flatten()
            r = {k: v.ravel() for k, v in result.to_dict().items()}
            r.update(
                {
                    "frequency[Hz]": freqs,
                    "amplitude[dimensionless]": amps,
                    "qubit": len(freqs) * [qubit],
                    "iteration": len(freqs) * [iteration],
                }
            )
            data.add_data_from_dict(r)

        # save data
        return data
        # TODO: calculate and save fit


def _fit(data):
    pass


def _plot(data: ResonatorPunchoutData, fit: ResonatorPunchoutResults, qubit):
    pass


resonator_punchout = Routine(_acquisition, _fit, _plot)
