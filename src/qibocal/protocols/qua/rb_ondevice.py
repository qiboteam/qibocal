import time
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import mpld3
import numpy as np
import numpy.typing as npt
from qibolab.platform import Platform
from qibolab.qubits import QubitId
from qm import generate_qua_script
from qm.qua import *  # nopycln: import
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from qualang_tools.results import fetching_tool
from scipy.optimize import curve_fit

from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.protocols.utils import table_dict, table_html

from .configuration import generate_config
from .utils import RBType, filter_function, filter_term, generate_depths, power_law

# parser.add_argument("--simulation-duration", type=int, default=None)
# parser.add_argument("--relaxation-time", type=int, default=None)


@dataclass
class RbOnDeviceParameters(Parameters):
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


def generate_sequence(max_circuit_depth, seed):
    sequence = declare(int, size=max_circuit_depth + 1)
    i = declare(int)
    rand = Random(seed=seed)

    with for_(i, 0, i < max_circuit_depth, i + 1):
        assign(sequence[i], rand.rand_int(24))

    return sequence


def generate_sequence_with_inverse(max_circuit_depth, seed, c1_table, inv_gates):
    cayley = declare(int, value=c1_table.flatten().tolist())
    inv_list = declare(int, value=inv_gates)
    current_state = declare(int)
    step = declare(int)
    sequence = declare(int, size=max_circuit_depth + 1)
    inv_gate = declare(int, size=max_circuit_depth + 1)
    i = declare(int)
    rand = Random(seed=seed)

    assign(current_state, 0)
    with for_(i, 0, i < max_circuit_depth, i + 1):
        assign(step, rand.rand_int(24))
        assign(current_state, cayley[current_state * 24 + step])
        assign(sequence[i], step)
        assign(inv_gate[i], inv_list[current_state])

    return sequence, inv_gate


def play_sequence(qubit, sequence_list, apply_inverse, depth, rx_duration):
    i = declare(int)
    condition = (i <= depth) if apply_inverse else (i < depth)
    with for_(i, 0, condition, i + 1):
        with switch_(sequence_list[i], unsafe=True):
            with case_(0):
                wait(rx_duration // 4, qubit)
            with case_(1):
                play("x180", qubit)
            with case_(2):
                play("y180", qubit)
            with case_(3):
                play("y180", qubit)
                play("x180", qubit)
            with case_(4):
                play("x90", qubit)
                play("y90", qubit)
            with case_(5):
                play("x90", qubit)
                play("-y90", qubit)
            with case_(6):
                play("-x90", qubit)
                play("y90", qubit)
            with case_(7):
                play("-x90", qubit)
                play("-y90", qubit)
            with case_(8):
                play("y90", qubit)
                play("x90", qubit)
            with case_(9):
                play("y90", qubit)
                play("-x90", qubit)
            with case_(10):
                play("-y90", qubit)
                play("x90", qubit)
            with case_(11):
                play("-y90", qubit)
                play("-x90", qubit)
            with case_(12):
                play("x90", qubit)
            with case_(13):
                play("-x90", qubit)
            with case_(14):
                play("y90", qubit)
            with case_(15):
                play("-y90", qubit)
            with case_(16):
                play("-x90", qubit)
                play("y90", qubit)
                play("x90", qubit)
            with case_(17):
                play("-x90", qubit)
                play("-y90", qubit)
                play("x90", qubit)
            with case_(18):
                play("x180", qubit)
                play("y90", qubit)
            with case_(19):
                play("x180", qubit)
                play("-y90", qubit)
            with case_(20):
                play("y180", qubit)
                play("x90", qubit)
            with case_(21):
                play("y180", qubit)
                play("-x90", qubit)
            with case_(22):
                play("x90", qubit)
                play("y90", qubit)
                play("x90", qubit)
            with case_(23):
                play("-x90", qubit)
                play("y90", qubit)
                play("-x90", qubit)


RbOnDeviceType = np.dtype(
    [
        ("state", np.int32),
        ("sequences", np.int32),
    ]
)


@dataclass
class RbOnDeviceData(Data):
    rb_type: str
    relaxation_time: int
    depths: list[int]
    data: dict[QubitId, dict[str, npt.NDArray[np.int32]]]

    def _get_data(self, key: str):
        qubit = self.qubits[0]
        try:
            arrays = self.data[qubit].item(0)
            return arrays[key]
        except AttributeError:
            return self.data[qubit][key]

    @property
    def state(self):
        return self._get_data("state")

    @property
    def sequences(self):
        return self._get_data("sequences")


def _acquisition(
    params: RbOnDeviceParameters, platform: Platform, targets: list[QubitId]
) -> RbOnDeviceData:
    assert len(targets) == 1
    target = targets[0]
    qubit = f"drive{target}"
    resonator = f"readout{target}" if target is not None else f"readout{target}"
    save_sequences = params.save_sequences
    apply_inverse = params.apply_inverse
    relaxation_time = params.relaxation_time

    num_of_sequences = params.num_of_sequences
    n_avg = params.n_avg
    max_circuit_depth = params.max_circuit_depth
    delta_clifford = params.delta_clifford
    assert (
        max_circuit_depth / delta_clifford
    ).is_integer(), "max_circuit_depth / delta_clifford must be an integer."
    seed = params.seed
    state_discrimination = params.state_discrimination
    # List of recovery gates from the lookup table
    inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]

    ###################
    # The QUA program #
    ###################
    with program() as rb:
        depth = declare(int)  # QUA variable for the varying depth
        depth_target = declare(
            int
        )  # QUA variable for the current depth (changes in steps of delta_clifford)
        # QUA variable to store the last Clifford gate of the current sequence which is replaced by the recovery gate
        saved_gate = declare(int)
        m = declare(int)  # QUA variable for the loop over random sequences
        n = declare(int)  # QUA variable for the averaging loop
        I = declare(fixed)  # QUA variable for the 'I' quadrature
        Q = declare(fixed)  # QUA variable for the 'Q' quadrature
        state = declare(bool)  # QUA variable for state discrimination
        # The relevant streams
        m_st = declare_stream()
        if state_discrimination:
            state_st = declare_stream()
        else:
            I_st = declare_stream()
            Q_st = declare_stream()
        # save random sequences to return to host
        if save_sequences:
            sequence_st = declare_stream()

        with for_(
            m, 0, m < num_of_sequences, m + 1
        ):  # QUA for_ loop over the random sequences
            if apply_inverse:
                sequence_list, inv_gate_list = generate_sequence_with_inverse(
                    max_circuit_depth, seed, c1_table, inv_gates
                )  # Generate the random sequence of length max_circuit_depth
            else:
                sequence_list = generate_sequence(max_circuit_depth, seed)

            if delta_clifford == 1:
                assign(depth_target, 1)
            else:
                assign(depth_target, 0)  # Initialize the current depth to 0

            if save_sequences:
                save(sequence_list[0], sequence_st)  # save sequence indices in stream
            with for_(
                depth, 1, depth <= max_circuit_depth, depth + 1
            ):  # Loop over the depths
                if save_sequences:
                    save(sequence_list[depth], sequence_st)
                # Replacing the last gate in the sequence with the sequence's inverse gate
                # The original gate is saved in 'saved_gate' and is being restored at the end
                assign(saved_gate, sequence_list[depth])
                if apply_inverse:
                    assign(sequence_list[depth], inv_gate_list[depth - 1])
                # Only played the depth corresponding to target_depth
                with if_((depth == 1) | (depth == depth_target)):
                    with for_(n, 0, n < n_avg, n + 1):  # Averaging loop
                        # Can be replaced by active reset
                        if relaxation_time > 0:
                            wait(relaxation_time // 4, resonator)
                        # Align the two elements to play the sequence after qubit initialization
                        align(resonator, qubit)
                        # The strict_timing ensures that the sequence will be played without gaps
                        with strict_timing_():
                            # Play the random sequence of desired depth
                            rx_duration = platform.qubits[
                                target
                            ].native_gates.RX.duration
                            play_sequence(
                                qubit, sequence_list, apply_inverse, depth, rx_duration
                            )
                        # Align the two elements to measure after playing the circuit.
                        align(qubit, resonator)
                        # wait(3 * (depth + 1) * (DRIVE_DURATION // 4), resonator)
                        # Make sure you updated the ge_threshold and angle if you want to use state discrimination
                        # state, I, Q = readout_macro(threshold=ge_threshold, state=state, I=I, Q=Q)
                        measure(
                            "measure",
                            resonator,
                            None,
                            dual_demod.full("cos", "out1", "sin", "out2", I),
                            dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                        )
                        # Save the results to their respective streams
                        if state_discrimination:
                            threshold = platform.qubits[target].threshold
                            iq_angle = platform.qubits[target].iq_angle
                            cos = np.cos(iq_angle)
                            sin = np.sin(iq_angle)
                            assign(state, I * cos - Q * sin > threshold)
                            save(state, state_st)
                        else:
                            save(I, I_st)
                            save(Q, Q_st)
                    # Go to the next depth
                    if params.logarithmic:
                        nmul = declare(int)
                        with for_(nmul, 0, nmul < delta_clifford, nmul + 1):
                            assign(depth_target, 2 * depth_target)
                    else:
                        assign(depth_target, depth_target + delta_clifford)
                # Reset the last gate of the sequence back to the original Clifford gate
                # (that was replaced by the recovery gate at the beginning)
                assign(sequence_list[depth], saved_gate)
            # Save the counter for the progress bar
            save(m, m_st)

        depths = generate_depths(max_circuit_depth, delta_clifford, params.logarithmic)
        with stream_processing():
            ndepth = len(depths)
            m_st.save("iteration")
            if save_sequences:
                sequence_st.buffer(max_circuit_depth + 1).buffer(num_of_sequences).save(
                    "sequences"
                )

            if state_discrimination:
                if n_avg > 1:
                    # saves a 2D array of depth and random pulse sequences in order to get error bars along the random sequences
                    state_st.boolean_to_int().buffer(n_avg).map(
                        FUNCTIONS.average()
                    ).buffer(ndepth).buffer(num_of_sequences).save("state")
                    # returns a 1D array of averaged random pulse sequences vs depth of circuit for live plotting
                    state_st.boolean_to_int().buffer(n_avg).map(
                        FUNCTIONS.average()
                    ).buffer(ndepth).average().save("state_avg")
                else:
                    # saves a 2D array of depth and random pulse sequences in order to get error bars along the random sequences
                    state_st.boolean_to_int().buffer(ndepth).buffer(
                        num_of_sequences
                    ).save("state")
                    # returns a 1D array of averaged random pulse sequences vs depth of circuit for live plotting
                    state_st.boolean_to_int().buffer(ndepth).average().save("state_avg")
            else:
                I_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(ndepth).buffer(
                    num_of_sequences
                ).save("I")
                Q_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(ndepth).buffer(
                    num_of_sequences
                ).save("Q")
                I_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(
                    ndepth
                ).average().save("I_avg")
                Q_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(
                    ndepth
                ).average().save("Q_avg")

    # Print total relaxation time (estimate of execution time)
    total_relaxation = relaxation_time * num_of_sequences * n_avg * ndepth * 1e-9
    print("\nTotal relaxation time: %.2f sec\n" % total_relaxation)

    #####################################
    #  Open Communication with the QOP  #
    #####################################
    controller = platform._controller
    qmm = controller.manager

    ###########################
    # Run or Simulate Program #
    ###########################
    # Open the quantum machine
    config = generate_config(platform, list(platform.qubits.keys()))
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(rb)

    if params.debug:
        with open("qua_script.py", "w") as file:
            file.write(generate_qua_script(rb, config))

    # Get results from QUA program
    if state_discrimination:
        results = fetching_tool(job, data_list=["state_avg", "iteration"], mode="live")
    else:
        results = fetching_tool(
            job, data_list=["I_avg", "Q_avg", "iteration"], mode="live"
        )

    start_time = time.time()
    while results.is_processing():
        # data analysis
        if state_discrimination:
            state_avg, iteration = results.fetch_all()
            value_avg = state_avg
        else:
            I, Q, iteration = results.fetch_all()
            value_avg = I
    final_time = time.time()

    # At the end of the program, fetch the non-averaged results to get the error-bars
    rb_type = RBType.infer(apply_inverse, relaxation_time).value
    if state_discrimination:
        if save_sequences:
            results = fetching_tool(job, data_list=["state", "sequences"])
            state, sequences = results.fetch_all()
            data = {target: {"state": state, "sequences": sequences}}
        else:
            results = fetching_tool(job, data_list=["state"])
            state = results.fetch_all()[0]
            data = {target: {"state": state}}
    else:
        raise NotImplementedError
        # if save_sequences:
        #    results = fetching_tool(job, data_list=["I", "Q", "sequences"])
        #    I, Q, sequences = results.fetch_all()
        #    data.voltage_i[target] = I
        #    data.voltage_q[target] = Q
        #    data.sequences[target] = sequences
        # else:
        #    results = fetching_tool(job, data_list=["I", "Q"])
        #    I, Q = results.fetch_all()
        #    data.voltage_i[target] = I
        #    data.voltage_q[target] = Q
    return RbOnDeviceData(
        rb_type=rb_type,
        relaxation_time=relaxation_time,
        depths=[int(x) for x in depths],
        data=data,
    )


@dataclass
class RbOnDeviceResults(Results):
    ydata: dict[QubitId, list[float]] = field(default_factory=dict)
    ysigma: dict[QubitId, list[float]] = field(default_factory=dict)
    pars: dict[QubitId, list[float]] = field(default_factory=dict)
    cov: dict[QubitId, list[float]] = field(default_factory=dict)


def process_data(data: RbOnDeviceData):
    rb_type = RBType(data.rb_type)
    depths = data.depths
    state = data.state

    if rb_type is RBType.STANDARD:
        return 1 - np.mean(state, axis=0), np.std(state, axis=0) / np.sqrt(
            state.shape[0]
        )

    is_restless = rb_type is RBType.RESTLESS
    term = filter_term(depths, state, data.sequences, is_restless=is_restless)
    ff = filter_function(term)
    return np.mean(ff, axis=1), np.std(ff, axis=1) / np.sqrt(ff.shape[1])


def _fit(data: RbOnDeviceData) -> RbOnDeviceResults:
    qubit = data.qubits[0]

    ydata, ysigma = process_data(data)
    results = RbOnDeviceResults(
        ydata={qubit: list(ydata)}, ysigma={qubit: list(ysigma)}
    )
    try:
        pars, cov = curve_fit(
            f=power_law,
            xdata=data.depths,
            ydata=ydata,
            sigma=ysigma,
            p0=[0.5, 0.0, 0.9],
            bounds=(-np.inf, np.inf),
            maxfev=2000,
        )
        results.pars[qubit] = list(pars)
        results.cov[qubit] = list(cov.flatten())
    except RuntimeError:
        pass
    return results


def _plot(data: RbOnDeviceData, target: QubitId, fit: RbOnDeviceResults):
    depths = data.depths
    state = data.state

    if fit is not None:
        ydata = fit.ydata[target]
        ysigma = fit.ysigma[target]
    else:
        ydata, ysigma = process_data(data)

    fitting_report = table_html(
        table_dict(
            target,
            [
                "RB type",
                "Number of sequences",
                "Relaxation time (us)",
            ],
            [
                (data.rb_type.capitalize(),),
                (len(state),),
                (data.relaxation_time // 1000,),
            ],
        )
    )

    pars = None
    if fit is not None:
        pars = fit.pars.get(target)
    if pars is not None:
        stdevs = np.sqrt(np.diag(np.reshape(fit.cov[target], (3, 3))))
        one_minus_p = 1 - pars[2]
        r_c = one_minus_p * (1 - 1 / 2**1)
        r_g = r_c / 1.875  # 1.875 is the average number of gates in clifford operation
        r_c_std = stdevs[2] * (1 - 1 / 2**1)
        r_g_std = r_c_std / 1.875
        fitting_report = "\n".join(
            [
                fitting_report,
                table_html(
                    table_dict(
                        target,
                        [
                            "A",
                            "B",
                            "p",
                            "Error rate (1-p)",
                            "Clifford set infidelity",
                            "Gate infidelity",
                        ],
                        [
                            (np.round(pars[0], 3), np.round(stdevs[0], 3)),
                            (np.round(pars[1], 3), np.round(stdevs[1], 3)),
                            (np.round(pars[2], 3), np.round(stdevs[2], 3)),
                            (one_minus_p, stdevs[2]),
                            (r_c, r_c_std),
                            (r_g, r_g_std),
                        ],
                        display_error=True,
                    )
                ),
            ]
        )

    fig = plt.figure(figsize=(16, 6))
    title = f"{data.rb_type.capitalize()} RB"
    plt.errorbar(
        depths, ydata, ysigma, marker="o", linestyle="-", markersize=4, label="data"
    )
    if pars is not None:
        max_circuit_depth = depths[-1]
        x = np.linspace(0, max_circuit_depth + 0.1, 1000)
        plt.plot(x, power_law(x, *pars), linestyle="--", label="fit")
    plt.xlabel("Depth")
    plt.ylabel("Survival probability")
    plt.legend()

    figures = [mpld3.fig_to_html(fig)]
    return figures, fitting_report


def _update(results: RbOnDeviceResults, platform: Platform, target: QubitId):
    pass


rb_ondevice = Routine(_acquisition, _fit, _plot, _update)
