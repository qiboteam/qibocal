import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot


@plot(
    "MSR and Phase vs Resonator Frequency and TWPA Frequency",
    plots.twpa_frequency,
)
def twpa_cal_frequency(
    platform: AbstractPlatform,
    qubits: dict,
    freq_width,
    freq_step,
    twpa_freq_min,
    twpa_freq_max,
    twpa_freq_step,
    relaxation_time=50,
    nshots=1024,
    software_averages=1,
):
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
    delta_frequency_range = np.arange(-freq_width // 2, freq_width // 2, freq_step)
    freq_sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        [ro_pulses[qubit] for qubit in qubits],
    )

    # twpa frequency
    twpa_frequency_range = np.arange(twpa_freq_min, twpa_freq_max, twpa_freq_step)

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include resonator frequency and attenuation
    data = DataUnits(
        name=f"data",
        quantities={"frequency": "Hz", "twpa_frequency": "Hz"},
        options=["qubit", "iteration"],
    )

    # repeat the experiment as many times as defined by software_averages
    for iteration in range(software_averages):
        for twpa_frequency in twpa_frequency_range:
            platform.instruments["twpa_pump"].frequency = twpa_frequency
            results = platform.sweep(
                sequence,
                freq_sweeper,
                nshots=nshots,
                relaxation_time=relaxation_time,
            )

            # retrieve the results for every qubit
            for qubit, ro_pulse in ro_pulses.items():
                # average msr, phase, i and q over the number of shots defined in the runcard
                result = results[ro_pulse.serial]
                # store the results
                freqs = delta_frequency_range + ro_pulse.frequency
                r = result.raw
                r.update(
                    {
                        "frequency[Hz]": freqs,
                        "twpa_frequency[Hz]": len(freqs) * [twpa_frequency],
                        "qubit": len(freqs) * [qubit],
                        "iteration": len(freqs) * [iteration],
                    }
                )
                data.add_data_from_dict(r)

            # save data
            yield data
        # TODO: calculate and save fit


@plot(
    "MSR and Phase vs Resonator Frequency and TWPA Power",
    plots.twpa_power,
)
def twpa_cal_power(
    platform: AbstractPlatform,
    qubits: dict,
    freq_width,
    freq_step,
    twpa_pow_min,
    twpa_pow_max,
    twpa_pow_step,
    relaxation_time=50,
    nshots=1024,
    software_averages=1,
):
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
    delta_frequency_range = np.arange(-freq_width // 2, freq_width // 2, freq_step)
    freq_sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        [ro_pulses[qubit] for qubit in qubits],
    )

    # twpa power
    twpa_power_range = np.arange(twpa_pow_min, twpa_pow_max, twpa_pow_step)

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include resonator frequency and power
    data = DataUnits(
        name=f"data",
        quantities={"frequency": "Hz", "twpa_power": "dB"},
        options=["qubit", "iteration"],
    )

    # repeat the experiment as many times as defined by software_averages
    for iteration in range(software_averages):
        for twpa_power in twpa_power_range:
            platform.instruments["twpa_pump"].power = twpa_power
            results = platform.sweep(
                sequence,
                freq_sweeper,
                nshots=nshots,
                relaxation_time=relaxation_time,
            )

            # retrieve the results for every qubit
            for qubit, ro_pulse in ro_pulses.items():
                # average msr, phase, i and q over the number of shots defined in the runcard
                result = results[ro_pulse.serial]
                # store the results
                freqs = delta_frequency_range + ro_pulse.frequency
                r = result.raw
                r.update(
                    {
                        "frequency[Hz]": freqs,
                        "twpa_power[dB]": len(freqs) * [twpa_power],
                        "qubit": len(freqs) * [qubit],
                        "iteration": len(freqs) * [iteration],
                    }
                )
                data.add_data_from_dict(r)

            # save data
            yield data
        # TODO: calculate and save fit
