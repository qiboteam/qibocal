from dataclasses import dataclass
from typing import Optional

import mpld3
import numpy as np
from qibolab import Platform, VirtualZ
from qm.qua import align, assign, declare, dual_demod, fixed, measure, save, wait
from qualang_tools.bakery.bakery import Baking

from qibocal.auto.operation import Parameters, QubitId, QubitPairId, Results, Routine

from .configuration import generate_config
from .two_qubit_rb import QuaTwoQubitRbData, TwoQubitRb


@dataclass
class QuaTwoQubitRbParameters(Parameters):
    circuit_depths: list[int]
    """How many consecutive Clifford gates within one executed circuit

    (https://qiskit.org/documentation/apidoc/circuit.html)
    """
    num_circuits_per_depth: int
    """How many random circuits within one depth."""
    num_shots_per_circuit: int
    """Repetitions of the same circuit (averaging)."""
    force_offsets: bool = False
    """Force sweetspot offsets using ``set_dc_offset`` command in the QUA program."""
    debug: Optional[str] = None
    """Dump QUA script and config in a file with this name."""


def _acquisition(
    params: QuaTwoQubitRbParameters, platform: Platform, targets: list[QubitPairId]
) -> QuaTwoQubitRbData:
    assert len(targets) == 1
    qubit1, qubit2 = targets[0]
    try:
        qubit1 = int(qubit1)
        qubit2 = int(qubit2)
    except ValueError:
        pass

    native_cz = platform.natives.two_qubit[(qubit1, qubit2)].CZ
    flux_channel = native_cz[0][0]
    if flux_channel != platform.qubits[qubit1].flux:
        if flux_channel == platform.qubits[qubit2].flux:
            qubit1, qubit2 = qubit2, qubit1
        else:
            raise ValueError("Flux pulse on different qubit not supported.")

    ##############################
    ## General helper functions ##
    ##############################
    def prep():
        # thermal preparation in clock cycles
        wait(params.relaxation_time // 4)
        align()

    def multiplexed_readout(i, i_st, q, q_st, qubits):
        """Perform multiplexed readout on two resonators"""
        for ind, qb in enumerate(qubits):
            measure(
                "measure",
                f"readout{qb}",
                None,
                dual_demod.full("cos", "out1", "sin", "out2", i[ind]),
                dual_demod.full("minus_sin", "out1", "cos", "out2", q[ind]),
            )
            if i_st is not None:
                save(i[ind], i_st[ind])
            if q_st is not None:
                save(q[ind], q_st[ind])

    def discriminate(target, i, q, state):
        acquisition_config = platform.config(platform.qubits[target].acquisition)
        cos = np.cos(acquisition_config.iq_angle)
        sin = np.sin(acquisition_config.iq_angle)
        assign(state, i * cos - q * sin > acquisition_config.threshold)

    def meas():
        I1 = declare(fixed)
        I2 = declare(fixed)
        Q1 = declare(fixed)
        Q2 = declare(fixed)
        state1 = declare(bool)
        state2 = declare(bool)
        # readout macro for multiplexed readout
        multiplexed_readout([I1, I2], None, [Q1, Q2], None, qubits=[qubit1, qubit2])
        discriminate(qubit1, I1, Q1, state1)
        discriminate(qubit2, I2, Q2, state2)
        return state1, state2

    ##############################
    ##  Two-qubit RB functions  ##
    ##############################
    # single qubit generic gate constructor Z^{z}Z^{a}X^{x}Z^{-a}
    # that can reach any point on the Bloch sphere (starting from arbitrary points)
    def bake_phased_xz(baker: Baking, q, x, z, a):
        if q == 1:
            element = f"drive{qubit1}"
        else:
            element = f"drive{qubit2}"

        baker.frame_rotation_2pi(a / 2, element)
        baker.play("x180", element, amp=x)
        baker.frame_rotation_2pi(-(a + z) / 2, element)

    # single qubit phase corrections in units of 2pi applied after the CZ gate
    vz1 = next(native_cz.channel(platform.qubits[qubit1].drive))
    vz2 = next(native_cz.channel(platform.qubits[qubit2].drive))
    assert isinstance(vz1, VirtualZ)
    assert isinstance(vz2, VirtualZ)
    qubit1_frame_update = vz1.phase / (2 * np.pi)
    qubit2_frame_update = vz2.phase / (2 * np.pi)

    # defines the CZ gate that realizes the mapping |00> -> |00>, |01> -> |01>, |10> -> |10>, |11> -> -|11>
    def bake_cz(baker: Baking, q1, q2):
        q1_xy_element = f"drive{qubit1}"
        q2_xy_element = f"drive{qubit2}"
        q1_z_element = f"flux{qubit1}"

        baker.play("cz", q1_z_element)
        baker.align()
        baker.frame_rotation_2pi(qubit1_frame_update, q1_xy_element)
        baker.frame_rotation_2pi(qubit2_frame_update, q2_xy_element)
        baker.align()

    ##############################
    ##  Two-qubit RB execution  ##
    ##############################
    # create RB experiment from configuration and defined functions
    config = generate_config(platform, platform.qubits.keys(), targets=[qubit1, qubit2])

    qmm = platform._controller.manager
    # for debugging when there is an error
    # from qm import generate_qua_script
    # from qm.qua import program
    # with open("rb2q_qua_config.py", "w") as file:
    #    with program() as prog:
    #        align()
    #    file.write(generate_qua_script(prog, config))

    rb = TwoQubitRb(
        config=config,
        single_qubit_gate_generator=bake_phased_xz,
        two_qubit_gate_generators={
            "CZ": bake_cz
        },  # can also provide e.g. "CNOT": bake_cnot
        prep_func=prep,
        measure_func=meas,
        interleaving_gate=None,
        # interleaving_gate=[cirq.CZ(cirq.LineQubit(0), cirq.LineQubit(1))],
        verify_generation=False,
    )

    offsets = (
        [
            (qb.flux, platform.config(qb.flux).offset)
            for qb in platform.qubits.values()
            if qb.flux is not None
        ]
        if params.force_offsets
        else []
    )
    data = rb.run(
        qmm,
        circuit_depths=params.circuit_depths,
        num_circuits_per_depth=params.num_circuits_per_depth,
        num_shots_per_circuit=params.num_shots_per_circuit,
        offsets=offsets,
        debug=params.debug,
    )

    # verify/save the random sequences created during the experiment
    # rb.save_sequences_to_file(
    #     "sequences.txt"
    # )  # saves the gates used in each random sequence
    # rb.save_command_mapping_to_file(
    #     "commands.txt"
    # )  # saves mapping from "command id" to sequence
    # rb.print_sequence()
    # rb.print_command_mapping()
    # rb.verify_sequences()  # simulates random sequences to ensure they recover to ground state. takes a while...
    return data


@dataclass
class QuaTwoQubitRbResults(Results):
    pass


def _fit(data: QuaTwoQubitRbData) -> QuaTwoQubitRbResults:
    return QuaTwoQubitRbResults()


def _plot(data: QuaTwoQubitRbData, target: QubitId, fit: QuaTwoQubitRbResults):
    fitting_report = ""

    figures = [
        data.plot_hist(n_cols=6, figsize=(12, len(data.circuit_depths) / 2)),
        data.plot_with_fidelity(figsize=(12, 6)),
    ]

    figures = [mpld3.fig_to_html(fig) for fig in figures]
    return figures, fitting_report


# # get the interleaved gate fidelity
# from two_qubit_rb.RBResult import get_interleaved_gate_fidelity
# interleaved_gate_fidelity = get_interleaved_gate_fidelity(
#     num_qubits=2,
#     reference_alpha=0.12345,  # replace with value from prior, non-interleaved experiment
#     # interleaved_alpha=res.fit_exponential()[1],  # alpha from the interleaved experiment
# )
# print(f"Interleaved Gate Fidelity: {interleaved_gate_fidelity*100:.3f}")


def _update(results: QuaTwoQubitRbResults, platform: Platform, target: QubitId):
    pass


qua_standard_rb_2q = Routine(_acquisition, _fit, _plot, _update)
