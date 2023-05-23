import numpy as np
from qibolab.executionparameters import AveragingMode
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
    qubits: dict,
    beta_param=None,
    software_averages=1,
    points=10,
):
    r"""
    The AllXY experiment is a simple test of the calibration of single qubit gatesThe qubit (initialized in the |0> state)
    is subjected to two back-to-back single-qubit gates and measured. In each round, we run 21 different gate pairs:
    ideally, the first 5 return the qubit to |0>, the next 12 drive it to superposition state, and the last 4 put the
    qubit in |1> state.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (dict): Dict of target Qubit objects to perform the action
        beta_param (float): Drag pi pulse coefficient. If none, teh default shape defined in the runcard will be used.
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Returns:
        A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Difference between resonator signal voltage mesurement in volts from sequence 1 and 2
            - **i[V]**: Difference between resonator signal voltage mesurement for the component I in volts from sequence 1 and 2
            - **q[V]**: Difference between resonator signal voltage mesurement for the component Q in volts from sequence 1 and 2
            - **phase[rad]**: Difference between resonator signal phase mesurement in radians from sequence 1 and 2
            - **probability[dimensionless]**: Probability of being in |0> state
            - **gateNumber[dimensionless]**: Gate number applied from the list of gates
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages
    """
    # reload instrument settings from runcard
    platform.reload_settings()

    # create a Data object to store the results
    data = Data(
        name="data",
        quantities={"probability", "gateNumber", "qubit", "iteration"},
    )

    count = 0
    # repeat the experiment as many times as defined by software_averages
    for iteration in range(software_averages):
        gateNumber = 1
        # sweep the parameter
        for gateNumber, gates in enumerate(gatelist):
            # save data as often as defined by points
            if count % points == 0 and count > 0:
                # save data
                yield data

            # create a sequence of pulses
            ro_pulses = {}
            sequence = PulseSequence()
            for qubit in qubits:
                sequence, ro_pulses[qubit] = _add_gate_pair_pulses_to_sequence(
                    platform, gates, qubit, beta_param, sequence
                )

            # execute the pulse sequence
            results = platform.execute_pulse_sequence(
                sequence, averaging_mode=AveragingMode.CYCLIC
            )

            # retrieve the results for every qubit
            for ro_pulse in ro_pulses.values():
                z_proj = 2 * results[ro_pulse.serial].ground_state_probability - 1
                # store the results
                r = {
                    "probability": z_proj,
                    "gateNumber": gateNumber,
                    "beta_param": beta_param,
                    "qubit": ro_pulse.qubit,
                    "iteration": iteration,
                }
                data.add(r)
            count += 1
    # finally, save the remaining data
    yield data


@plot("Probability vs Gate Sequence", plots.allXY_drag_pulse_tuning)
def allXY_drag_pulse_tuning(
    platform: AbstractPlatform,
    qubits: dict,
    beta_start,
    beta_end,
    beta_step,
    software_averages=1,
    points=10,
):
    r"""
    The AllXY experiment is a simple test of the calibration of single qubit gatesThe qubit (initialized in the |0> state)
    is subjected to two back-to-back single-qubit gates and measured. In each round, we run 21 different gate pairs:
    ideally, the first 5 return the qubit to |0>, the next 12 drive it to superposition state, and the last 4 put the
    qubit in |1> state.

    The AllXY iteration method allows the user to execute iteratively the list of gates playing with the drag pulse shape
    in order to find the optimal drag pulse coefficient for pi pulses.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (dict): Dict of target Qubit objects to perform the action
        beta_start (float): Initial drag pulse beta parameter
        beta_end (float): Maximum drag pulse beta parameter
        beta_step (float): Scan range step for the drag pulse beta parameter
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Returns:
        A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Difference between resonator signal voltage mesurement in volts from sequence 1 and 2
            - **i[V]**: Difference between resonator signal voltage mesurement for the component I in volts from sequence 1 and 2
            - **q[V]**: Difference between resonator signal voltage mesurement for the component Q in volts from sequence 1 and 2
            - **phase[rad]**: Difference between resonator signal phase mesurement in radians from sequence 1 and 2
            - **probability[dimensionless]**: Probability of being in |0> state
            - **gateNumber[dimensionless]**: Gate number applied from the list of gates
            - **beta_param[dimensionless]**: Beta paramter applied in the current execution
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages

    """

    # reload instrument settings from runcard
    platform.reload_settings()

    data = Data(
        name="data",
        quantities={"probability", "gateNumber", "beta_param", "qubit", "iteration"},
    )

    # repeat the experiment as many times as defined by software_averages
    count = 0
    for iteration in range(software_averages):
        # sweep the parameters
        for beta_param in np.arange(beta_start, beta_end, beta_step).round(4):
            gateNumber = 1
            for gates in gatelist:
                # save data as often as defined by points
                if count % points == 0 and count > 0:
                    # save data
                    yield data

                # create a sequence of pulses
                ro_pulses = {}
                sequence = PulseSequence()
                for qubit in qubits:
                    sequence, ro_pulses[qubit] = _add_gate_pair_pulses_to_sequence(
                        platform, gates, qubit, beta_param, sequence
                    )

                # execute the pulse sequence
                results = platform.execute_pulse_sequence(
                    sequence, averaging_mode=AveragingMode.CYCLIC
                )

                # retrieve the results for every qubit
                for ro_pulse in ro_pulses.values():
                    z_proj = 2 * results[ro_pulse.serial].ground_state_probability - 1
                    # store the results
                    r = {
                        "probability": z_proj,
                        "gateNumber": gateNumber,
                        "beta_param": beta_param,
                        "qubit": ro_pulse.qubit,
                        "iteration": iteration,
                    }
                    data.add(r)
                count += 1
                gateNumber += 1
    # finally, save the remaining data
    yield data


@plot("MSR vs beta parameter", plots.drag_pulse_tuning)
def drag_pulse_tuning(
    platform: AbstractPlatform,
    qubits: dict,
    beta_start,
    beta_end,
    beta_step,
    software_averages=1,
    points=10,
):
    r"""
    In this experiment, we apply two sequences in a given qubit: Rx(pi/2) - Ry(pi) and Ry(pi) - Rx(pi/2) for a range
    of different beta parameter values. After fitting, we obtain the best coefficient value for a pi pulse with drag shape.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (dict): Dict of target Qubit objects to perform the action
        beta_start (float): Initial drag pulse beta parameter
        beta_end (float): Maximum drag pulse beta parameter
        beta_step (float): Scan range step for the drag pulse beta parameter
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Returns:
        - A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Difference between resonator signal voltage mesurement in volts from sequence 1 and 2
            - **i[V]**: Difference between resonator signal voltage mesurement for the component I in volts from sequence 1 and 2
            - **q[V]**: Difference between resonator signal voltage mesurement for the component Q in volts from sequence 1 and 2
            - **phase[rad]**: Difference between resonator signal phase mesurement in radians from sequence 1 and 2
            - **beta_param[dimensionless]**: Optimal drag coefficient
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages

        - A DataUnits object with the fitted data obtained with the following keys

            - **optimal_beta_param**: Best drag pulse coefficent
            - **popt0**: offset
            - **popt1**: oscillation amplitude
            - **popt2**: period
            - **popt3**: phase
            - **qubit**: The qubit being tested
    """

    # reload instrument settings from runcard
    platform.reload_settings()

    # define the parameter to sweep and its range:
    # qubit drive DRAG pulse beta parameter
    beta_param_range = np.arange(beta_start, beta_end, beta_step).round(4)

    # create a DataUnits object to store the MSR, phase, i, q and the beta parameter
    data = DataUnits(
        name="data",
        quantities={"beta_param": "dimensionless"},
        options=["qubit", "iteration"],
    )

    count = 0
    # repeat the experiment as many times as defined by software_averages
    for iteration in range(software_averages):
        for beta_param in beta_param_range:
            # save data as often as defined by points
            if count % points == 0 and count > 0:
                # save data
                yield data
                # calculate and save fit
                yield drag_tuning_fit(
                    data,
                    x="beta_param[dimensionless]",
                    y="MSR[uV]",
                    qubits=qubits,
                    labels=["optimal_beta_param"],
                )

            # create two sequences of pulses
            # seq1: RX(pi/2) - RY(pi) - MZ
            # seq1: RY(pi/2) - RX(pi) - MZ

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

            # execute the pulse sequences
            result1 = platform.execute_pulse_sequence(
                seq1, averaging_mode=AveragingMode.CYCLIC
            )
            result2 = platform.execute_pulse_sequence(
                seq2, averaging_mode=AveragingMode.CYCLIC
            )

            # retrieve the results for every qubit
            for ro_pulse in ro_pulses.values():
                r1 = result1[ro_pulse.serial]
                r2 = result2[ro_pulse.serial]
                # store the results
                r = {
                    "MSR[V]": r1.measurement.mean() - r2.measurement.mean(),
                    "i[V]": r1.i.mean() - r2.i.mean(),
                    "q[V]": r1.q.mean() - r2.q.mean(),
                    "phase[rad]": r1.phase.mean() - r2.phase.mean(),
                    "beta_param[dimensionless]": beta_param,
                    "qubit": ro_pulse.qubit,
                    "iteration": iteration,
                }
                data.add(r)
            count += 1

    yield data
    yield drag_tuning_fit(
        data,
        x="beta_param[dimensionless]",
        y="MSR[uV]",
        qubits=qubits,
        labels=["optimal_beta_param"],
    )


def _add_gate_pair_pulses_to_sequence(
    platform: AbstractPlatform, gates, qubit, beta_param, sequence
):
    pulse_duration = platform.create_RX_pulse(qubit, start=0).duration
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
