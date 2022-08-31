# -*- coding: utf-8 -*-
import numpy as np
from qibolab.pulses import PulseSequence

from qcvv.calibrations.utils import variable_resolution_scanrange
from qcvv.data import Dataset
from qcvv.decorators import store


@store
def allXY(
    platform,
    qubit,
    beta_param=None,
    software_averages=1,
    points=10,
):
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

    gnd = complex(platform.characterization["single_qubit"][qubit]["state1_voltage"])
    exc = complex(platform.characterization["single_qubit"][qubit]["state0_voltage"])
    data = Dataset(
        name=f"data_q{qubit}", quantities={"probability": "dimensionless", "gateNumber": "dimensionless"}
    )

    count = 0
    for _ in range(software_averages):
        gateNumber = 1
        for gates in gatelist:
            if count % points == 0:
                yield data
            seq, ro_pulse = _get_sequence_from_gate_pair(
                platform, gates, qubit, beta_param
            )
            seq.add(ro_pulse)
            msr, phase, i, q = platform.execute_pulse_sequence(seq, nshots=1024)[
                ro_pulse.serial
            ]
            prob = np.abs(msr * 1e6 - gnd) / (exc - gnd)
            prob = (2 * prob) - 1
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[rad]": phase,
                "probability[dimensionless]": prob,
                "gateNumber[dimensionless]": np.array(gateNumber),
            }
            data.add(results)
            count += 1
            gateNumber += 1
    yield data


@store
def allXY_iteration(
    platform,
    qubit,
    beta_start,
    beta_end,
    beta_step,
    software_averages=1,
    points=10,
):
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

    gnd = complex(platform.characterization["single_qubit"][qubit]["state1_voltage"])
    exc = complex(platform.characterization["single_qubit"][qubit]["state0_voltage"])
    data = Dataset(
        name=f"data_q{qubit}",
        quantities={"probability": "", "gateNumber": "dimensionless", "beta_param": "dimensionless"},
    )

    count = 0
    for _ in range(software_averages):
        for beta_param in np.arange(beta_start, beta_end, beta_step).round(1):
            gateNumber = 1
            for gates in gatelist:
                if count % points == 0:
                    yield data
                seq, ro_pulse = _get_sequence_from_gate_pair(
                    platform, gates, qubit, beta_param
                )
                seq.add(ro_pulse)
                msr, phase, i, q = platform.execute_pulse_sequence(seq, nshots=1024)[
                    ro_pulse.serial
                ]
                prob = np.abs(msr * 1e6 - gnd) / (exc - gnd)
                prob = (2 * prob) - 1
                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "probability[dimensionless]": prob,
                    "gateNumber[dimensionless]": np.array(gateNumber),
                    "beta_param[dimensionless]": np.array(beta_param),
                }
                data.add(results)
                count += 1
                gateNumber += 1
    yield data


def _get_sequence_from_gate_pair(platform, gates, qubit, beta_param):
    sampling_rate = platform.sampling_rate
    pulse_frequency = platform.settings["native_gates"]["single_qubit"][qubit]["RX"][
        "frequency"
    ]
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
                RX90_pulse = platform.RX90_drag_pulse(
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
                RY90_pulse = platform.RX90_drag_pulse(
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
