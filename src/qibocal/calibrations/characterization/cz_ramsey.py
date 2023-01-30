import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import SNZ, FluxPulse, Pulse, PulseSequence, PulseType, Rectangular

from qibocal import plots
from qibocal.calibrations.characterization.utils import iq_to_prob
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qibocal.fitting import fit_amplitude_balance_cz


@plot("Phi2Q", plots.amplitude_balance_cz_phi2q)
@plot("Leakage", plots.amplitude_balance_cz_leakage)
@plot("Acquired phase", plots.amplitude_balance_cz_acquired_phase)
@plot("Raw data at 180 degree", plots.amplitude_balance_cz_raw_data)
def amplitude_balance_cz(
    platform: AbstractPlatform,
    qubits: dict,
    positive_amplitude_start,
    positive_amplitude_end,
    positive_amplitude_step,
    snz_ratio_amplitude_start,
    snz_ratio_amplitude_end,
    snz_ratio_amplitude_step,
    detuning_start,
    detuning_end,
    detuning_step,
    points=10,
):
    r"""
    This experiment is used to find the optimal amplitude for the flux pulse for
    CZ gate.

    Two sets of sequences are played:
    1. The first set of sequences is applying a RX90 pulse, a flux pulse, a RX90
    pulse and a readout pulse on the target qubit, while only applying a readout
    pulse on the control qubit.
    2. The second set of sequences is applying a RX90 pulse, a flux pulse, a
    RX90 pulse and a readout pulse on the target qubit, and a RX pulse and
    readout pulse on the control qubit.

    Each set of sequences is played while varying the phase of the second RX90
    pulse on the control qubit. This is a Ramsey like experiment which allows to
    find the detuning acquired by the flux pulse. The phase of both ON and OFF
    sequences are measured and the difference between the two is used to find
    the optimal amplitude (positive half of SNZ pulse) and optimal ratio
    (positive/negative half of SNZ pulse) that will minimize |02> leakage while
    performing CZ gate (180 phase difference).

    Args:
        platform: The platform to run the experiment on.
        qubit: The qubit to perform the experiment on.
        positive_amplitude_start: The starting amplitude of the positive half of
            the flux pulse.
        positive_amplitude_end: The ending amplitude of the positive half of the
            flux pulse.
        positive_amplitude_step: The step size of the positive half of the flux
            pulse.
        snz_ratio_amplitude_start: The starting ratio amplitude of the positive
            over the negative half of the SNZ flux pulse.
        snz_ratio_amplitude_end: The ending ratio amplitude of the positive over
            the negative half of the SNZ flux pulse.
        snz_ratio_amplitude_step: The step size of the ratio amplitude of the
            positive over the negative half of the SNZ flux pulse.
        detuning_start: The starting phase of the second RX90 pulse in degree to
            the target qubit.
        detuning_end: The ending phase of the second RX90 pulse in degree to the
            target qubit.
        detuning_step: The step size of the phase of the second RX90 pulse in
            degree to the target qubit.
        points: The number of points over which to yield the data.

    Returns:
        It yields the data of the experiment, and its fitting results.
    """

    platform.reload_settings()

    # Get the qubits to perform the experiment on
    qubit_control = {}
    qubit_target = {}
    for q in qubits:
        topology = platform.topology[platform.qubits.index(q)]
        for j in platform.qubits:
            if (
                topology[j] == 1
                and platform.qubits[j] != q
                and platform.qubits[j] in qubits
            ):
                if (
                    platform.characterization["single_qubit"][q]["qubit_freq"]
                    > platform.characterization["single_qubit"][platform.qubits[j]][
                        "qubit_freq"
                    ]
                ):
                    if (
                        platform.qubits[j]
                        not in np.array(qubit_control)[np.array(qubit_target) == q]
                    ):
                        qubit_control += [platform.qubits[j]]
                        qubit_target += [q]
                if (
                    platform.characterization["single_qubit"][q]["qubit_freq"]
                    < platform.characterization["single_qubit"][platform.qubits[j]][
                        "qubit_freq"
                    ]
                ):
                    if (
                        platform.qubits[j]
                        not in np.array(qubit_target)[np.array(qubit_control) == q]
                    ):
                        qubit_control += [q]
                        qubit_target += [platform.qubits[j]]

    data = DataUnits(
        name=f"data",
        quantities={
            "detuning": "degree",
            "flux_pulse_amplitude": "dimensionless",
            "flux_pulse_ratio": "dimensionless",
            "prob": "dimensionless",
        },
        options=["controlqubit", "targetqubit", "ON_OFF", "result_qubit"],
    )

    # Variables
    amplitudes = np.arange(
        positive_amplitude_start,
        positive_amplitude_end,
        positive_amplitude_step,
    )
    ratios = np.arange(
        snz_ratio_amplitude_start,
        snz_ratio_amplitude_end,
        snz_ratio_amplitude_step,
    )
    detuning = np.arange(
        detuning_start,
        detuning_end,
        detuning_step,
    )

    for i, q_target in enumerate(qubit_target):

        # Target sequence RX90 - CPhi - RX90 - MZ
        initial_RX90_pulse = platform.create_RX90_pulse(
            q_target, start=0, relative_phase=0
        )
        tp = 20
        flux_pulse = platform.create_CZ_pulse(
            [q_target, qubit_control[i]],
            start=initial_RX90_pulse.se_finish + 8,
        )
        RX90_pulse = platform.create_RX90_pulse(
            q_target, start=flux_pulse.finish + 8, relative_phase=0
        )
        ro_pulse_target = platform.create_qubit_readout_pulse(
            q_target, start=RX90_pulse.se_finish
        )

        # Creating different measurment
        sequence_target = initial_RX90_pulse + flux_pulse + RX90_pulse + ro_pulse_target

        # Control sequence
        initial_RX_pulse = platform.create_RX_pulse(qubit_control[i], start=0)
        RX_pulse = platform.create_RX_pulse(qubit_control[i], start=RX90_pulse.se_start)
        ro_pulse_control = platform.create_qubit_readout_pulse(
            qubit_control[i], start=RX90_pulse.se_finish
        )

        # Mean and excited states
        mean_gnd = {
            qubit_target[i]: complex(
                platform.characterization["single_qubit"][qubit_target[i]][
                    "mean_gnd_states"
                ]
            ),
            qubit_control[i]: complex(
                platform.characterization["single_qubit"][qubit_control[i]][
                    "mean_gnd_states"
                ]
            ),
        }
        mean_exc = {
            qubit_target[i]: complex(
                platform.characterization["single_qubit"][qubit_target[i]][
                    "mean_exc_states"
                ]
            ),
            qubit_control[i]: complex(
                platform.characterization["single_qubit"][qubit_control[i]][
                    "mean_exc_states"
                ]
            ),
        }

        count = 0
        for amplitude in amplitudes:
            for ratio in ratios:
                if count % points == 0 and count > 1:
                    yield data
                    # yield fit_amplitude_balance_cz(data)
                flux_pulse[0].amplitude = amplitude
                flux_pulse[1].amplitude = amplitude * ratio

                for det in detuning:
                    RX90_pulse.relative_phase = np.deg2rad(det)

                    # while True:  # FIXME: Long scan, it is to avoid QBlox bug
                    #     try:
                    sequenceON = (
                        sequence_target + initial_RX_pulse + RX_pulse + ro_pulse_control
                    )
                    sequenceOFF = sequence_target + ro_pulse_control

                    platform_results = platform.execute_pulse_sequence(sequenceON)

                    for ro_pulse in sequenceON.ro_pulses:
                        results = {
                            "MSR[V]": platform_results[ro_pulse.serial][0],
                            "i[V]": platform_results[ro_pulse.serial][2],
                            "q[V]": platform_results[ro_pulse.serial][3],
                            "phase[rad]": platform_results[ro_pulse.serial][1],
                            "prob[dimensionless]": iq_to_prob(
                                platform_results[ro_pulse.serial][2],
                                platform_results[ro_pulse.serial][3],
                                mean_gnd[ro_pulse.qubit],
                                mean_exc[ro_pulse.qubit],
                            ),
                            "controlqubit": qubit_control[i],
                            "targetqubit": qubit_target[i],
                            "result_qubit": ro_pulse.qubit,
                            "ON_OFF": "ON",
                            "detuning[degree]": det,
                            "flux_pulse_amplitude[dimensionless]": amplitude,
                            "flux_pulse_ratio[dimensionless]": ratio,
                        }
                        data.add(results)

                    platform_results = platform.execute_pulse_sequence(sequenceOFF)

                    for ro_pulse in sequenceOFF.ro_pulses:
                        results = {
                            "MSR[V]": platform_results[ro_pulse.serial][0],
                            "i[V]": platform_results[ro_pulse.serial][2],
                            "q[V]": platform_results[ro_pulse.serial][3],
                            "phase[rad]": platform_results[ro_pulse.serial][1],
                            "prob[dimensionless]": iq_to_prob(
                                platform_results[ro_pulse.serial][2],
                                platform_results[ro_pulse.serial][3],
                                mean_gnd[ro_pulse.qubit],
                                mean_exc[ro_pulse.qubit],
                            ),
                            "controlqubit": qubit_control[i],
                            "targetqubit": qubit_target[i],
                            "result_qubit": ro_pulse.qubit,
                            "ON_OFF": "OFF",
                            "detuning[degree]": det,
                            "flux_pulse_amplitude[dimensionless]": amplitude,
                            "flux_pulse_ratio[dimensionless]": ratio,
                        }
                        data.add(results)

                        # except:
                        #     continue
                        # break

                count += 1
            yield data


@plot("Phi2Q", plots.amplitude_balance_cz_phi2q)
@plot("Leakage", plots.amplitude_balance_cz_leakage)
@plot("Acquired phase", plots.amplitude_balance_cz_acquired_phase)
@plot("Raw data at 180 degree", plots.amplitude_balance_cz_raw_data)
def snz_tune_up(
    platform: AbstractPlatform,
    qubits: dict,
    positive_amplitude_start,
    positive_amplitude_end,
    positive_amplitude_step,
    snz_ratio_amplitude_start,
    snz_ratio_amplitude_end,
    snz_ratio_amplitude_step,
    detuning_start,
    detuning_end,
    detuning_step,
    points=10,
):
    r"""
    This experiment is used to find the optimal amplitude for the flux pulse for
    CZ gate.

    Two sets of sequences are played:
    1. The first set of sequences is applying a RX90 pulse, a flux pulse, a RX90
    pulse and a readout pulse on the target qubit, while only applying a readout
    pulse on the control qubit.
    2. The second set of sequences is applying a RX90 pulse, a flux pulse, a
    RX90 pulse and a readout pulse on the target qubit, and a RX pulse and
    readout pulse on the control qubit.

    Each set of sequences is played while varying the phase of the second RX90
    pulse on the control qubit. This is a Ramsey like experiment which allows to
    find the detuning acquired by the flux pulse. The phase of both ON and OFF
    sequences are measured and the difference between the two is used to find
    the optimal amplitude (positive half of SNZ pulse) and optimal ratio
    (positive/negative half of SNZ pulse) that will minimize |02> leakage while
    performing CZ gate (180 phase difference).

    Args:
        platform: The platform to run the experiment on.
        qubit: The qubit to perform the experiment on.
        positive_amplitude_start: The starting amplitude of the positive half of
            the flux pulse.
        positive_amplitude_end: The ending amplitude of the positive half of the
            flux pulse.
        positive_amplitude_step: The step size of the positive half of the flux
            pulse.
        snz_ratio_amplitude_start: The starting ratio amplitude of the positive
            over the negative half of the SNZ flux pulse.
        snz_ratio_amplitude_end: The ending ratio amplitude of the positive over
            the negative half of the SNZ flux pulse.
        snz_ratio_amplitude_step: The step size of the ratio amplitude of the
            positive over the negative half of the SNZ flux pulse.
        detuning_start: The starting phase of the second RX90 pulse in degree to
            the target qubit.
        detuning_end: The ending phase of the second RX90 pulse in degree to the
            target qubit.
        detuning_step: The step size of the phase of the second RX90 pulse in
            degree to the target qubit.
        points: The number of points over which to yield the data.

    Returns:
        It yields the data of the experiment, and its fitting results.
    """

    platform.reload_settings()

    # Get the qubits to perform the experiment on
    qubit_control = {}
    qubit_target = {}
    for q in qubits:
        topology = platform.topology[platform.qubits.index(q)]
        for j in platform.qubits:
            if (
                topology[j] == 1
                and platform.qubits[j] != q
                and platform.qubits[j] in qubits
            ):
                if (
                    platform.characterization["single_qubit"][q]["qubit_freq"]
                    > platform.characterization["single_qubit"][platform.qubits[j]][
                        "qubit_freq"
                    ]
                ):
                    if (
                        platform.qubits[j]
                        not in np.array(qubit_control)[np.array(qubit_target) == q]
                    ):
                        qubit_control += [platform.qubits[j]]
                        qubit_target += [q]
                if (
                    platform.characterization["single_qubit"][q]["qubit_freq"]
                    < platform.characterization["single_qubit"][platform.qubits[j]][
                        "qubit_freq"
                    ]
                ):
                    if (
                        platform.qubits[j]
                        not in np.array(qubit_target)[np.array(qubit_control) == q]
                    ):
                        qubit_control += [q]
                        qubit_target += [platform.qubits[j]]

    data = DataUnits(
        name=f"data",
        quantities={
            "detuning": "degree",
            "flux_pulse_amplitude": "dimensionless",
            "flux_pulse_ratio": "dimensionless",
            "prob": "dimensionless",
        },
        options=["controlqubit", "targetqubit", "ON_OFF", "result_qubit"],
    )

    # Variables
    amplitudes = np.arange(
        positive_amplitude_start,
        positive_amplitude_end,
        positive_amplitude_step,
    )
    ratios = np.arange(
        snz_ratio_amplitude_start,
        snz_ratio_amplitude_end,
        snz_ratio_amplitude_step,
    )
    detuning = np.arange(
        detuning_start,
        detuning_end,
        detuning_step,
    )

    for i, q_target in enumerate(qubit_target):

        # Target sequence RX90 - CPhi - RX90 - MZ
        initial_RX90_pulse = platform.create_RX90_pulse(
            q_target, start=0, relative_phase=0
        )
        tp = 20
        flux_pulse = platform.create_CZ_pulse(
            [q_target, qubit_control[i]],
            start=initial_RX90_pulse.se_finish + 8,
        )
        RX90_pulse = platform.create_RX90_pulse(
            q_target, start=flux_pulse.finish + 8, relative_phase=0
        )
        ro_pulse_target = platform.create_qubit_readout_pulse(
            q_target, start=RX90_pulse.se_finish
        )

        # Creating different measurment
        sequence_target = initial_RX90_pulse + flux_pulse + RX90_pulse + ro_pulse_target

        # Control sequence
        initial_RX_pulse = platform.create_RX_pulse(qubit_control[i], start=0)
        RX_pulse = platform.create_RX_pulse(qubit_control[i], start=RX90_pulse.se_start)
        ro_pulse_control = platform.create_qubit_readout_pulse(
            qubit_control[i], start=RX90_pulse.se_finish
        )

        # Mean and excited states
        mean_gnd = {
            qubit_target[i]: complex(
                platform.characterization["single_qubit"][qubit_target[i]][
                    "mean_gnd_states"
                ]
            ),
            qubit_control[i]: complex(
                platform.characterization["single_qubit"][qubit_control[i]][
                    "mean_gnd_states"
                ]
            ),
        }
        mean_exc = {
            qubit_target[i]: complex(
                platform.characterization["single_qubit"][qubit_target[i]][
                    "mean_exc_states"
                ]
            ),
            qubit_control[i]: complex(
                platform.characterization["single_qubit"][qubit_control[i]][
                    "mean_exc_states"
                ]
            ),
        }

        count = 0
        for amplitude in amplitudes:
            for ratio in ratios:
                if count % points == 0 and count > 1:
                    yield data
                    # yield fit_amplitude_balance_cz(data)
                flux_pulse[0].amplitude = amplitude
                flux_pulse[0].shape = SNZ(10, ratio, 0.9734188817598534)

                for det in detuning:
                    RX90_pulse.relative_phase = np.deg2rad(det)

                    # while True:  # FIXME: Long scan, it is to avoid QBlox bug
                    #     try:
                    sequenceON = (
                        sequence_target + initial_RX_pulse + RX_pulse + ro_pulse_control
                    )
                    sequenceOFF = sequence_target + ro_pulse_control

                    platform_results = platform.execute_pulse_sequence(sequenceON)

                    for ro_pulse in sequenceON.ro_pulses:
                        results = {
                            "MSR[V]": platform_results[ro_pulse.serial][0],
                            "i[V]": platform_results[ro_pulse.serial][2],
                            "q[V]": platform_results[ro_pulse.serial][3],
                            "phase[rad]": platform_results[ro_pulse.serial][1],
                            "prob[dimensionless]": iq_to_prob(
                                platform_results[ro_pulse.serial][2],
                                platform_results[ro_pulse.serial][3],
                                mean_gnd[ro_pulse.qubit],
                                mean_exc[ro_pulse.qubit],
                            ),
                            "controlqubit": qubit_control[i],
                            "targetqubit": qubit_target[i],
                            "result_qubit": ro_pulse.qubit,
                            "ON_OFF": "ON",
                            "detuning[degree]": det,
                            "flux_pulse_amplitude[dimensionless]": amplitude,
                            "flux_pulse_ratio[dimensionless]": ratio,
                        }
                        data.add(results)

                    platform_results = platform.execute_pulse_sequence(sequenceOFF)

                    for ro_pulse in sequenceOFF.ro_pulses:
                        results = {
                            "MSR[V]": platform_results[ro_pulse.serial][0],
                            "i[V]": platform_results[ro_pulse.serial][2],
                            "q[V]": platform_results[ro_pulse.serial][3],
                            "phase[rad]": platform_results[ro_pulse.serial][1],
                            "prob[dimensionless]": iq_to_prob(
                                platform_results[ro_pulse.serial][2],
                                platform_results[ro_pulse.serial][3],
                                mean_gnd[ro_pulse.qubit],
                                mean_exc[ro_pulse.qubit],
                            ),
                            "controlqubit": qubit_control[i],
                            "targetqubit": qubit_target[i],
                            "result_qubit": ro_pulse.qubit,
                            "ON_OFF": "OFF",
                            "detuning[degree]": det,
                            "flux_pulse_amplitude[dimensionless]": amplitude,
                            "flux_pulse_ratio[dimensionless]": ratio,
                        }
                        data.add(results)

                        # except:
                        #     continue
                        # break

                count += 1
            yield data


@plot("snz", plots.chevron_iswap)
def chevron_iswap(
    platform: AbstractPlatform,
    qubits: dict,
    flux_pulse_duration_start,
    flux_pulse_duration_end,
    flux_pulse_duration_step,
    flux_pulse_amplitude_start,
    flux_pulse_amplitude_end,
    flux_pulse_amplitude_step,
    points=10,
):
    # 1) from state |0> apply Rx(pi/2) to state |i>,
    # 2) apply a flux pulse of variable duration,
    # 3) measure in the X and Y axis
    #   MY = Rx(pi/2) - (flux)(t) - Rx(pi/2) - MZ
    #   MX = Rx(pi/2) - (flux)(t) - Ry(pi/2) - MZ
    # The flux pulse detunes the qubit and results in a rotation around the Z axis = atan(MY/MX)

    platform.reload_settings()

    qubit_control = []
    qubit_target = []
    for i, q in enumerate(platform.topology[platform.qubits.index(qubit)]):
        if q == 1 and platform.qubits[i] != qubit:
            if (
                platform.characterization["single_qubit"][qubit]["qubit_freq"]
                > platform.characterization["single_qubit"][platform.qubits[i]][
                    "qubit_freq"
                ]
            ):
                qubit_control += [platform.qubits[i]]
                qubit_target += [qubit]
            if (
                platform.characterization["single_qubit"][qubit]["qubit_freq"]
                < platform.characterization["single_qubit"][platform.qubits[i]][
                    "qubit_freq"
                ]
            ):
                qubit_control += [qubit]
                qubit_target += [platform.qubits[i]]

    data = DataUnits(
        name=f"data_q{qubit}",
        quantities={
            "flux_pulse_duration": "ns",
            "flux_pulse_amplitude": "dimensionless",
            "prob": "dimensionless",
        },
        options=["controlqubit", "targetqubit", "result_qubit"],
    )

    for i, q_target in enumerate(qubit_target):

        # Target sequence RX - iSWAP - MZ
        initial_RX_pulse = platform.create_RX_pulse(q_target, start=0)

        flux_pulse = FluxPulse(
            start=initial_RX_pulse.se_finish + 8,
            duration=flux_pulse_duration_start,  # 2 * flux_pulse_duration_start+ 4,  # sweep to produce oscillations [300 to 400ns] in steps od 1ns? or 4?
            amplitude=flux_pulse_amplitude_start,  # fix for each run
            relative_phase=0,
            shape=Rectangular(),  # SNZ(flux_pulse_duration_start),  # should be rectangular, but it gets distorted
            channel=platform.qubit_channel_map[q_target][2],
            qubit=q_target,
        )

        ro_pulse_target = platform.create_qubit_readout_pulse(
            q_target, start=flux_pulse.se_finish
        )

        sequence = initial_RX_pulse + flux_pulse + ro_pulse_target

        # # Control sequence - in case we do other way
        # initial_RX_pulse = platform.create_RX_pulse(qubit_control[i], start=0)
        # RX_pulse = platform.create_RX_pulse(qubit_control[i], start=RX90_pulse.se_start)
        # ro_pulse_control = platform.create_qubit_readout_pulse(
        #     qubit_control[i], start=RX90_pulse.se_finish
        # )

        # Variables
        amplitudes = np.arange(
            flux_pulse_amplitude_start,
            flux_pulse_amplitude_end,
            flux_pulse_amplitude_step,
        )
        durations = np.arange(
            flux_pulse_duration_start,
            flux_pulse_duration_end,
            flux_pulse_duration_step,
        )

        # Mean and excited states
        mean_gnd = {
            qubit_target[i]: complex(
                platform.characterization["single_qubit"][qubit_target[i]][
                    "mean_gnd_states"
                ]
            ),
            qubit_control[i]: complex(
                platform.characterization["single_qubit"][qubit_control[i]][
                    "mean_gnd_states"
                ]
            ),
        }
        mean_exc = {
            qubit_target[i]: complex(
                platform.characterization["single_qubit"][qubit_target[i]][
                    "mean_exc_states"
                ]
            ),
            qubit_control[i]: complex(
                platform.characterization["single_qubit"][qubit_control[i]][
                    "mean_exc_states"
                ]
            ),
        }

        count = 0
        for amplitude in amplitudes:
            for duration in durations:
                if count % points == 0:
                    yield data
                flux_pulse.amplitude = amplitude
                flux_pulse.duration = duration  # 2 * duration + 4
                # flux_pulse.shape = SNZ(duration)

                while True:
                    try:
                        platform_results = platform.execute_pulse_sequence(sequence)

                        for ro_pulse in sequence.ro_pulses:
                            results = {
                                "MSR[V]": platform_results[ro_pulse.serial][0],
                                "i[V]": platform_results[ro_pulse.serial][2],
                                "q[V]": platform_results[ro_pulse.serial][3],
                                "phase[rad]": platform_results[ro_pulse.serial][1],
                                "prob[dimensionless]": iq_to_prob(
                                    platform_results[ro_pulse.serial][2],
                                    platform_results[ro_pulse.serial][3],
                                    mean_gnd[ro_pulse.qubit],
                                    mean_exc[ro_pulse.qubit],
                                ),
                                "controlqubit": qubit_control[i],
                                "targetqubit": qubit_target[i],
                                "result_qubit": ro_pulse.qubit,
                                "flux_pulse_duration[ns]": duration,
                                "flux_pulse_amplitude[dimensionless]": amplitude,
                            }
                            data.add(results)

                    except:
                        continue
                    break

                count += 1
            yield data
