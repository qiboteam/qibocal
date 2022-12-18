import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.data import Data, DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import drag_tuning_fit

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


@plot("Probability vs Gate Sequence", plots.allXY)
def allXY(
    platform: AbstractPlatform,
    qubits: list,
    beta_param=None,
    software_averages=1,
    points=10,
):
    platform.reload_settings()

    data = Data(
        name="data",
        quantities={"probability", "gateNumber", "qubit"},
    )

    count = 0
    for _ in range(software_averages):
        gateNumber = 1
        for gates in gatelist:
            if count % points == 0 and count > 0:
                yield data

            ro_pulses = {}
            sequence = PulseSequence()
            for qubit in qubits:
                sequence, ro_pulses[qubit] = _add_gate_pair_pulses_to_sequence(
                    platform, gates, qubit, beta_param, sequence
                )

            results = platform.execute_pulse_sequence(sequence)
            for qubit in qubits:
                prob = 1 - 2 * results["probability"][ro_pulses[qubit].serial]

                r = {
                    "probability": prob,
                    "gateNumber": gateNumber,
                    "qubit": qubit,
                }
                data.add(r)
            count += 1
            gateNumber += 1
    yield data


@plot("Probability vs Gate Sequence", plots.allXY_drag_pulse_tuning)
def allXY_drag_pulse_tuning(
    platform: AbstractPlatform,
    qubits: list,
    beta_start,
    beta_end,
    beta_step,
    software_averages=1,
    points=10,
):
    platform.reload_settings()

    data = Data(
        name="data",
        quantities={"probability", "gateNumber", "beta_param", "qubit"},
    )

    count = 0
    for _ in range(software_averages):
        for beta_param in np.arange(beta_start, beta_end, beta_step).round(4):
            gateNumber = 1
            for gates in gatelist:
                if count % points == 0 and count > 0:
                    yield data

                ro_pulses = {}
                sequence = PulseSequence()
                for qubit in qubits:
                    sequence, ro_pulses[qubit] = _add_gate_pair_pulses_to_sequence(
                        platform, gates, qubit, beta_param, sequence
                    )

                results = platform.execute_pulse_sequence(sequence)
                for qubit in qubits:
                    prob = 1 - 2 * results["probability"][ro_pulses[qubit].serial]

                    r = {
                        "probability": prob,
                        "gateNumber": gateNumber,
                        "beta_param": beta_param,
                        "qubit": qubit,
                    }
                    data.add(r)
                count += 1
                gateNumber += 1
    yield data


@plot("MSR vs beta parameter", plots.msr_beta)
def drag_pulse_tuning(
    platform: AbstractPlatform,
    qubits: list,
    beta_start,
    beta_end,
    beta_step,
    software_averages=1,
    points=10,
):
    platform.reload_settings()

    data = DataUnits(
        name="data", quantities={"beta_param": "dimensionless"}, options=["qubit"]
    )

    count = 0
    for _ in range(software_averages):
        for beta_param in np.arange(beta_start, beta_end, beta_step).round(4):
            if count % points == 0 and count > 0:
                yield data
                for qubit in qubits:
                    yield drag_tuning_fit(
                        data.get_column("qubit", qubit),
                        x="beta_param[dimensionless]",
                        y="MSR[uV]",
                        qubit=qubit,
                        nqubits=platform.settings["nqubits"],
                        labels=[
                            "optimal_beta_param",
                        ],
                    )

            ro_pulses = {}
            seq1 = PulseSequence()
            seq2 = PulseSequence()
            for qubit in qubits:
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
                # drag pulse RY(pi/2)
                RY90_drag_pulse = platform.create_RX90_drag_pulse(
                    qubit, start=0, relative_phase=np.pi / 2, beta=beta_param
                )
                # drag pulse RX(pi)
                RX_drag_pulse = platform.create_RX_drag_pulse(
                    qubit, start=RY90_drag_pulse.finish, beta=beta_param
                )

                # RO pulse
                ro_pulses[qubit] = platform.create_qubit_readout_pulse(
                    qubit,
                    start=2
                    * RX90_drag_pulse.duration,  # assumes all single-qubit gates have same duration
                )
                # RX(pi/2) - RY(pi) - RO
                seq1.add(RX90_drag_pulse)
                seq1.add(RY_drag_pulse)
                seq1.add(ro_pulses[qubit])

                # RX(pi/2) - RY(pi) - RO
                seq2.add(RY90_drag_pulse)
                seq2.add(RX_drag_pulse)
                seq2.add(ro_pulses[qubit])

            result1 = platform.execute_pulse_sequence(seq1)
            result2 = platform.execute_pulse_sequence(seq2)

            for qubit in qubits:
                msr1, phase1, i1, q1 = result1[ro_pulses[qubit].serial]
                msr2, phase2, i2, q2 = result2[ro_pulses[qubit].serial]

                results = {
                    "MSR[V]": msr1 - msr2,
                    "i[V]": i1 - i2,
                    "q[V]": q1 - q2,
                    "phase[rad]": phase1 - phase2,
                    "beta_param[dimensionless]": beta_param,
                    "qubit": qubit,
                }
                data.add(results)
            count += 1

    yield data


def _add_gate_pair_pulses_to_sequence(
    platform: AbstractPlatform, gates, qubit, beta_param, sequence
):

    pulse_duration = platform.settings["native_gates"]["single_qubit"][qubit]["RX"][
        "duration"
    ]
    # All gates have equal pulse duration

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
    sequence.add(ro_pulse)
    return sequence, ro_pulse
