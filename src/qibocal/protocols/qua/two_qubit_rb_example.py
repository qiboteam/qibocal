import matplotlib.pyplot as plt
from qm import QuantumMachinesManager, qua
from qm.qua import align, assign, declare, dual_demod, save, wait
from qualang_tools.bakery.bakery import Baking
from two_qubit_rb import TwoQubitRb


@dataclass
class QuaRbTwoQubitParameters(Parameters):
    num_of_sequences: int
    max_circuit_depth: int
    "Maximum circuit depth"
    delta_clifford: int = 1
    "Play each sequence with a depth step equals to delta_clifford"
    logarithmic: bool = False
    """Use logarithmically scaled depths.

    The depths used in this case are 1, 2^delta_clifford, 2^(2*delta_clifford), ...
    up to the specified ``max_circuit_depth``.
    """
    seed: Optional[int] = None
    "Pseudo-random number generator seed"
    n_avg: int = 1
    "Number of averaging loops for each random sequence"
    save_sequences: bool = True
    apply_inverse: bool = False
    state_discrimination: bool = True
    "Flag to enable state discrimination if the readout has been calibrated (rotated blobs and threshold)"
    debug: bool = False
    "If enabled it dumps the qua script."

    def __post_init__(self):
        if self.seed is None:
            self.seed = np.random.randint(0, int(1e6))


def _acquisition(
    params: QuaRbTwoQubitParameters, platform: Platform, targets: list[QubitPairId]
):
    assert len(targets) == 1
    q1, q2 = targets[0]

    ##############################
    ## General helper functions ##
    ##############################
    def prep():
        # thermal preparation in clock cycles
        wait(params.relaxation_time // 4)
        align()

    def multiplexed_readout(
        I, I_st, Q, Q_st, resonators, sequential=False, amplitude=1.0, weights=""
    ):
        """Perform multiplexed readout on two resonators"""
        if type(resonators) is not list:
            resonators = [resonators]

        for ind, res in enumerate(resonators):
            measure(
                "readout" * qua.amp(amplitude),
                f"rr{res}",
                None,
                dual_demod.full(weights + "cos", weights + "sin", I[ind]),
                dual_demod.full(weights + "minus_sin", weights + "cos", Q[ind]),
            )

            if I_st is not None:
                save(I[ind], I_st[ind])
            if Q_st is not None:
                save(Q[ind], Q_st[ind])

            if sequential and ind < len(resonators) - 1:
                align(f"rr{res}", f"rr{res+1}")

    def meas():
        threshold1 = (
            0.3  # threshold for state discrimination 0 <-> 1 using the I quadrature
        )
        threshold2 = (
            0.3  # threshold for state discrimination 0 <-> 1 using the I quadrature
        )
        I1 = declare(fixed)
        I2 = declare(fixed)
        Q1 = declare(fixed)
        Q2 = declare(fixed)
        state1 = declare(bool)
        state2 = declare(bool)
        multiplexed_readout(
            [I1, I2], None, [Q1, Q2], None, resonators=[1, 2], weights="rotated_"
        )  # readout macro for multiplexed readout
        assign(state1, I1 > threshold1)  # assume that all information is in I
        assign(state2, I2 > threshold2)  # assume that all information is in I
        return state1, state2

    ##############################
    ##  Two-qubit RB functions  ##
    ##############################
    # assign a string to a variable to be able to call them in the functions
    q1_idx_str = "1"
    q2_idx_str = "2"

    # single qubit generic gate constructor Z^{z}Z^{a}X^{x}Z^{-a}
    # that can reach any point on the Bloch sphere (starting from arbitrary points)
    def bake_phased_xz(baker: Baking, q, x, z, a):
        if q == 1:
            element = f"q{q1_idx_str}_xy"
        else:
            element = f"q{q2_idx_str}_xy"

        baker.frame_rotation_2pi(a / 2, element)
        baker.play("x180", element, amp=x)
        baker.frame_rotation_2pi(-(a + z) / 2, element)

    # single qubit phase corrections in units of 2pi applied after the CZ gate
    qubit1_frame_update = 0.23  # example values, should be taken from QPU parameters
    qubit2_frame_update = 0.12  # example values, should be taken from QPU parameters

    # defines the CZ gate that realizes the mapping |00> -> |00>, |01> -> |01>, |10> -> |10>, |11> -> -|11>
    def bake_cz(baker: Baking, q1, q2):
        q1_xy_element = f"q{q1_idx_str}_xy"
        q2_xy_element = f"q{q2_idx_str}_xy"
        q1_z_element = f"q{q1_idx_str}_z"

        baker.play("cz", q1_z_element)
        baker.align()
        baker.frame_rotation_2pi(qubit1_frame_update, q1_xy_element)
        baker.frame_rotation_2pi(qubit2_frame_update, q2_xy_element)
        baker.align()


##############################
##  Two-qubit RB execution  ##
##############################
qmm = QuantumMachinesManager(
    host=qop_ip, port=qop_port, cluster_name=cluster_name
)  # initialize qmm

# create RB experiment from configuration and defined functions
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

res = rb.run(
    qmm,
    circuit_depths=[1, 2, 3, 4, 5],
    num_circuits_per_depth=2,
    num_shots_per_circuit=1,
)

# circuit_depths ~ how many consecutive Clifford gates within one executed circuit
# (https://qiskit.org/documentation/apidoc/circuit.html)
# num_circuits_per_depth ~ how many random circuits within one depth
# num_shots_per_circuit ~ repetitions of the same circuit (averaging)

res.plot_hist()
plt.show()

res.plot_fidelity()
plt.show()

# verify/save the random sequences created during the experiment
rb.save_sequences_to_file(
    "sequences.txt"
)  # saves the gates used in each random sequence
rb.save_command_mapping_to_file(
    "commands.txt"
)  # saves mapping from "command id" to sequence
# rb.print_sequence()
# rb.print_command_mapping()
# rb.verify_sequences()  # simulates random sequences to ensure they recover to ground state. takes a while...

# # get the interleaved gate fidelity
# from two_qubit_rb.RBResult import get_interleaved_gate_fidelity
# interleaved_gate_fidelity = get_interleaved_gate_fidelity(
#     num_qubits=2,
#     reference_alpha=0.12345,  # replace with value from prior, non-interleaved experiment
#     # interleaved_alpha=res.fit_exponential()[1],  # alpha from the interleaved experiment
# )
# print(f"Interleaved Gate Fidelity: {interleaved_gate_fidelity*100:.3f}")
