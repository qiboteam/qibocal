from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, Parameters, Results, Routine


@dataclass
class AllXYParameters(Parameters):
    """AllXY runcard inputs."""

    beta_param: float = None
    """Beta parameter for drag pulse."""
    unrolling: bool = False
    """If ``True`` it uses sequence unrolling to deploy multiple sequences in a single instrument call.
    Defaults to ``False``."""


@dataclass
class AllXYResults(Results):
    """AllXY outputs."""


AllXYType = np.dtype([("prob", np.float64), ("gate", "<U5"), ("errors", np.float64)])


@dataclass
class AllXYData(Data):
    """AllXY acquisition outputs."""

    beta_param: float = None
    """Beta parameter for drag pulse."""
    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""


gatelist = [
    ["I", "I"],
    ["Xp", "Xp"],
    ["Yp", "Yp"],
    ["Xp", "Yp"],
    ["Yp", "Xp"],
    ["X9", "I"],
    ["Y9", "I"],
    ["X9", "Y9"],
    ["Y9", "X9"],
    ["X9", "Yp"],
    ["Y9", "Xp"],
    ["Xp", "Y9"],
    ["Yp", "X9"],
    ["X9", "Xp"],
    ["Xp", "X9"],
    ["Y9", "Yp"],
    ["Yp", "Y9"],
    ["Xp", "I"],
    ["Yp", "I"],
    ["X9", "X9"],
    ["Y9", "Y9"],
]


def _acquisition(
    params: AllXYParameters,
    platform: Platform,
    targets: list[QubitId],
) -> AllXYData:
    r"""
    Data acquisition for allXY experiment.
    The AllXY experiment is a simple test of the calibration of single qubit gatesThe qubit (initialized in the |0> state)
    is subjected to two back-to-back single-qubit gates and measured. In each round, we run 21 different gate pairs:
    ideally, the first 5 return the qubit to |0>, the next 12 drive it to superposition state, and the last 4 put the
    qubit in |1> state.
    """

    # create a Data object to store the results
    data = AllXYData(beta_param=params.beta_param)

    # repeat the experiment as many times as defined by software_averages
    # for iteration in range(params.software_averages):
    sequences, all_ro_pulses = [], []
    for gates in gatelist:
        sequences.append(PulseSequence())
        all_ro_pulses.append({})
        for qubit in targets:
            sequences[-1], all_ro_pulses[-1][qubit] = add_gate_pair_pulses_to_sequence(
                platform, gates, qubit, sequences[-1], beta_param=params.beta_param
            )

    # execute the pulse sequence
    options = ExecutionParameters(
        nshots=params.nshots, averaging_mode=AveragingMode.CYCLIC
    )
    if params.unrolling:
        results = platform.execute_pulse_sequences(sequences, options)
    else:
        results = [
            platform.execute_pulse_sequence(sequence, options) for sequence in sequences
        ]

    for ig, (gates, ro_pulses) in enumerate(zip(gatelist, all_ro_pulses)):
        gate = "-".join(gates)
        for qubit in targets:
            serial = ro_pulses[qubit].serial
            if params.unrolling:
                prob = results[serial][ig].probability(0)
                z_proj = 2 * prob - 1
            else:
                prob = results[ig][serial].probability(0)
                z_proj = 2 * prob - 1

            errors = 2 * np.sqrt(prob * (1 - prob) / params.nshots)
            data.register_qubit(
                AllXYType,
                (qubit),
                dict(
                    prob=np.array([z_proj]),
                    gate=np.array([gate]),
                    errors=np.array([errors]),
                ),
            )

    # finally, save the remaining data
    return data


def add_gate_pair_pulses_to_sequence(
    platform: Platform,
    gates,
    qubit,
    sequence,
    sequence_delay=0,
    readout_delay=0,
    beta_param=None,
):
    pulse_duration = platform.create_RX_pulse(qubit, start=0).duration
    # All gates have equal pulse duration

    sequence_duration = sequence.get_qubit_pulses(qubit).duration + sequence_delay
    pulse_start = sequence.get_qubit_pulses(qubit).duration + sequence_delay

    for gate in gates:
        if gate == "I":
            pass

        if gate == "Xp":
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

        if gate == "X9":
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

        if gate == "Yp":
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

        if gate == "Y9":
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

        sequence_duration += pulse_duration
        pulse_start = sequence_duration

    # RO pulse starting just after pair of gates
    ro_pulse = platform.create_qubit_readout_pulse(
        qubit, start=sequence_duration + readout_delay
    )

    sequence.add(ro_pulse)
    return sequence, ro_pulse


def _fit(_data: AllXYData) -> AllXYResults:
    """Post-Processing for allXY"""
    return AllXYResults()


# allXY
def _plot(data: AllXYData, target: QubitId, fit: AllXYResults = None):
    """Plotting function for allXY."""

    figures = []
    fitting_report = ""
    fig = go.Figure()

    qubit_data = data[target]
    error_bars = qubit_data.errors
    probs = qubit_data.prob
    gates = qubit_data.gate

    fig.add_trace(
        go.Scatter(
            x=gates,
            y=probs,
            error_y=dict(
                type="data",
                array=error_bars,
                visible=True,
            ),
            mode="markers",
            text=gatelist,
            textposition="bottom center",
            name="Expectation value",
            showlegend=True,
            legendgroup="group1",
        ),
    )

    fig.add_hline(
        y=0,
        line_width=2,
        line_dash="dash",
        line_color="grey",
    )
    fig.add_hline(
        y=1,
        line_width=2,
        line_dash="dash",
        line_color="grey",
    )

    fig.add_hline(
        y=-1,
        line_width=2,
        line_dash="dash",
        line_color="grey",
    )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Gate sequence number",
        yaxis_title="Expectation value of Z",
    )

    figures.append(fig)

    return figures, fitting_report


allxy = Routine(_acquisition, _fit, _plot)
"""AllXY Routine object."""
