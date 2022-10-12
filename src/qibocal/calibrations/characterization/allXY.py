# -*- coding: utf-8 -*-
import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.data import Dataset
from qibocal.decorators import plot
from qibocal.fitting.methods import drag_tunning_fit

# allXY rotations
gatelist = [
    ["I", "I"],
    ["RX(pi)", "RX(pi)"],
    ["RY(pi)", "RY(pi)"],
    ["RX(pi)", "RY(pi)"],
    ["RY(pi)", "RX(pi)"],
    ["RX(pi/2)", "I"],
    ["RY(pi/2)", "I"],
    ["RX(pi/2)", "RY(pi/2)"],
    ["RY(pi/2)", "RX(pi/2)"],
    ["RX(pi/2)", "RY(pi)"],
    ["RY(pi/2)", "RX(pi)"],
    ["RX(pi)", "RY(pi/2)"],
    ["RY(pi)", "RX(pi/2)"],
    ["RX(pi/2)", "RX(pi)"],
    ["RX(pi)", "RX(pi/2)"],
    ["RY(pi/2)", "RY(pi)"],
    ["RY(pi)", "RY(pi/2)"],
    ["RX(pi)", "I"],
    ["RY(pi)", "I"],
    ["RX(pi/2)", "RX(pi/2)"],
    ["RY(pi/2)", "RY(pi/2)"],
]


@plot("Prob vs gate sequence", plots.prob_gate)
def allXY(
    platform: AbstractPlatform,
    qubit: int,
    beta_param=None,
    software_averages=1,
    points=10,
):

    state0_voltage = complex(
        platform.characterization["single_qubit"][qubit]["state0_voltage"]
    )
    state1_voltage = complex(
        platform.characterization["single_qubit"][qubit]["state1_voltage"]
    )

    data = Dataset(
        name=f"data_q{qubit}",
        quantities={"probability": "dimensionless", "gateNumber": "dimensionless"},
    )

    # FIXME: Waiting to be able to pass qpucard to qibolab
    ro_pulse_test = platform.create_qubit_readout_pulse(qubit, start=4)
    platform.ro_port[qubit].lo_frequency = (
        platform.characterization["single_qubit"][qubit]["resonator_freq"]
        - ro_pulse_test.frequency
    )

    qd_pulse_test = platform.create_qubit_drive_pulse(qubit, start=0, duration=4)
    platform.qd_port[qubit].lo_frequency = (
        platform.characterization["single_qubit"][qubit]["qubit_freq"]
        - qd_pulse_test.frequency
    )

    count = 0
    for _ in range(software_averages):
        gateNumber = 1
        for gates in gatelist:
            if count % points == 0 and count > 0:
                yield data
            seq, ro_pulse = _get_sequence_from_gate_pair(
                platform, gates, qubit, beta_param
            )
            seq.add(ro_pulse)
            msr, phase, i, q = platform.execute_pulse_sequence(seq, nshots=2048)[
                ro_pulse.serial
            ]

            prob = np.abs(msr * 1e6 - state1_voltage) / np.abs(
                state1_voltage - state0_voltage
            )
            prob = (2 * prob) - 1

            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[rad]": phase,
                "probability[dimensionless]": prob,
                "gateNumber[dimensionless]": gateNumber,
            }
            data.add(results)
            count += 1
            gateNumber += 1
    yield data


@plot("Prob vs gate sequence", plots.prob_gate_iteration)
def allXY_iteration(
    platform: AbstractPlatform,
    qubit: int,
    beta_start,
    beta_end,
    beta_step,
    software_averages=1,
    points=10,
):

    # FIXME: Waiting to be able to pass qpucard to qibolab
    ro_pulse_test = platform.create_qubit_readout_pulse(qubit, start=4)
    platform.ro_port[qubit].lo_frequency = (
        platform.characterization["single_qubit"][qubit]["resonator_freq"]
        - ro_pulse_test.frequency
    )

    qd_pulse_test = platform.create_qubit_drive_pulse(qubit, start=0, duration=4)
    platform.qd_port[qubit].lo_frequency = (
        platform.characterization["single_qubit"][qubit]["qubit_freq"]
        - qd_pulse_test.frequency
    )

    state0_voltage = complex(
        platform.characterization["single_qubit"][qubit]["state0_voltage"]
    )
    state1_voltage = complex(
        platform.characterization["single_qubit"][qubit]["state1_voltage"]
    )

    data = Dataset(
        name=f"data_q{qubit}",
        quantities={
            "probability": "dimensionless",
            "gateNumber": "dimensionless",
            "beta_param": "dimensionless",
        },
    )

    count = 0
    for _ in range(software_averages):
        for beta_param in np.arange(beta_start, beta_end, beta_step).round(4):
            gateNumber = 1
            for gates in gatelist:
                if count % points == 0 and count > 0:
                    yield data
                seq, ro_pulse = _get_sequence_from_gate_pair(
                    platform, gates, qubit, beta_param
                )
                seq.add(ro_pulse)
                msr, phase, i, q = platform.execute_pulse_sequence(seq, nshots=1024)[
                    ro_pulse.serial
                ]

                prob = np.abs(msr * 1e6 - state1_voltage) / np.abs(
                    state1_voltage - state0_voltage
                )
                prob = (2 * prob) - 1

                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "probability[dimensionless]": prob,
                    "gateNumber[dimensionless]": gateNumber,
                    "beta_param[dimensionless]": beta_param,
                }
                data.add(results)
                count += 1
                gateNumber += 1
    yield data


@plot("MSR vs beta parameter", plots.msr_beta)
def drag_pulse_tunning(
    platform: AbstractPlatform,
    qubit: int,
    beta_start,
    beta_end,
    beta_step,
    points=10,
):

    # platform.reload_settings()

    # FIXME: Waiting to be able to pass qpucard to qibolab
    ro_pulse_test = platform.create_qubit_readout_pulse(qubit, start=4)
    platform.ro_port[qubit].lo_frequency = (
        platform.characterization["single_qubit"][qubit]["resonator_freq"]
        - ro_pulse_test.frequency
    )

    qd_pulse_test = platform.create_qubit_drive_pulse(qubit, start=0, duration=4)
    platform.qd_port[qubit].lo_frequency = (
        platform.characterization["single_qubit"][qubit]["qubit_freq"]
        - qd_pulse_test.frequency
    )

    data = Dataset(name=f"data_q{qubit}", quantities={"beta_param": "dimensionless"})

    count = 0
    for beta_param in np.arange(beta_start, beta_end, beta_step).round(4):
        if count % points == 0 and count > 0:
            yield data
            yield drag_tunning_fit(
                data,
                x="beta_param[dimensionless]",
                y="MSR[uV]",
                qubit=qubit,
                nqubits=platform.settings["nqubits"],
                labels=[
                    "optimal_beta_param",
                ],
            )
        # drag pulse RX(pi/2)
        RX90_drag_pulse = platform.create_RX90_drag_pulse(
            qubit, start=0, beta=beta_param
        )
        # drag pulse RY(pi)
        RY_drag_pulse = platform.create_RX_drag_pulse(
            qubit,
            start=RX90_drag_pulse.finish,
            relative_phase=+np.pi / 2,
            beta=beta_param,
        )
        # RO pulse
        ro_pulse = platform.create_qubit_readout_pulse(
            qubit, start=RY_drag_pulse.finish
        )

        # Rx(pi/2) - Ry(pi) - Ro
        seq1 = PulseSequence()
        seq1.add(RX90_drag_pulse)
        seq1.add(RY_drag_pulse)
        seq1.add(ro_pulse)
        msr1, i1, q1, phase1 = platform.execute_pulse_sequence(seq1)[ro_pulse.serial]

        # drag pulse RY(pi/2)
        RY90_drag_pulse = platform.create_RX90_drag_pulse(
            qubit, start=0, relative_phase=np.pi / 2, beta=beta_param
        )
        # drag pulse RX(pi)
        RX_drag_pulse = platform.create_RX_drag_pulse(
            qubit, start=RY90_drag_pulse.finish, beta=beta_param
        )

        # Ry(pi/2) - Rx(pi) - Ro
        seq2 = PulseSequence()
        seq2.add(RY90_drag_pulse)
        seq2.add(RX_drag_pulse)
        seq2.add(ro_pulse)
        msr2, phase2, i2, q2 = platform.execute_pulse_sequence(seq2)[ro_pulse.serial]
        results = {
            "MSR[V]": msr1 - msr2,
            "i[V]": i1 - i2,
            "q[V]": q1 - q2,
            "phase[deg]": phase1 - phase2,
            "beta_param[dimensionless]": beta_param,
        }
        data.add(results)
        count += 1

    yield data


def _get_sequence_from_gate_pair(platform: AbstractPlatform, gates, qubit, beta_param):

    pulse_duration = platform.settings["native_gates"]["single_qubit"][qubit]["RX"][
        "duration"
    ]
    # All gates have equal pulse duration

    sequence = PulseSequence()

    sequenceDuration = 0
    pulse_start = 0

    for gate in gates:
        if gate == "I":
            # print("Transforming to sequence I gate")
            pass

        if gate == "RX(pi)":
            # print("Transforming to sequence RX(pi) gate")
            if beta_param == None:
                RX_pulse = platform.create_RX_pulse(
                    qubit,
                    start=pulse_start,
                )
            else:
                RX_pulse = platform.create_RX_drag_pulse(
                    qubit,
                    start=pulse_start,
                    beta=beta_param,
                )
            sequence.add(RX_pulse)

        if gate == "RX(pi/2)":
            # print("Transforming to sequence RX(pi/2) gate")
            if beta_param == None:
                RX90_pulse = platform.create_RX90_pulse(
                    qubit,
                    start=pulse_start,
                )
            else:
                RX90_pulse = platform.create_RX90_drag_pulse(
                    qubit,
                    start=pulse_start,
                    beta=beta_param,
                )
            sequence.add(RX90_pulse)

        if gate == "RY(pi)":
            # print("Transforming to sequence RY(pi) gate")
            if beta_param == None:
                RY_pulse = platform.create_RX_pulse(
                    qubit,
                    start=pulse_start,
                    relative_phase=np.pi / 2,
                )
            else:
                RY_pulse = platform.create_RX_drag_pulse(
                    qubit,
                    start=pulse_start,
                    relative_phase=np.pi / 2,
                    beta=beta_param,
                )
            sequence.add(RY_pulse)

        if gate == "RY(pi/2)":
            # print("Transforming to sequence RY(pi/2) gate")
            if beta_param == None:
                RY90_pulse = platform.create_RX90_pulse(
                    qubit,
                    start=pulse_start,
                    relative_phase=np.pi / 2,
                )
            else:
                RY90_pulse = platform.create_RX90_drag_pulse(
                    qubit,
                    start=pulse_start,
                    relative_phase=np.pi / 2,
                    beta=beta_param,
                )
            sequence.add(RY90_pulse)

        sequenceDuration = sequenceDuration + pulse_duration
        pulse_start = pulse_duration

    # RO pulse starting just after pair of gates
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=sequenceDuration + 4)

    return sequence, ro_pulse
