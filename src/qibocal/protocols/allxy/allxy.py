from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Drag,
    Pulse,
    PulseSequence,
)

from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.update import replace

__all__ = ["allxy", "gatelist", "AllXYType", "allxy_sequence"]


@dataclass
class AllXYParameters(Parameters):
    """AllXY runcard inputs."""

    beta_param: float = None
    """Beta parameter for drag pulse. If None is given, the native rx pulse in the parameters will be used"""
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
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> AllXYData:
    r"""
    Data acquisition for allXY experiment.
    The AllXY experiment is a simple test of the calibration of single qubit gatesThe qubit (initialized in the 0 state)
    is subjected to two back-to-back single-qubit gates and measured. In each round, we run 21 different gate pairs:
    ideally, the first 5 return the qubit to 0, the next 12 drive it to superposition state, and the last 4 put the
    qubit in 1 state.
    """

    # create a Data object to store the results
    data = AllXYData(beta_param=params.beta_param)

    sequences, all_ro_pulses = [], []
    for gates in gatelist:
        sequence = PulseSequence()
        ro_pulses = {}
        for qubit in targets:
            qubit_sequence, ro_pulses[qubit] = allxy_sequence(
                platform, gates, qubit, beta_param=params.beta_param
            )
            sequence += qubit_sequence
        sequences.append(sequence)
        all_ro_pulses.append(ro_pulses)

    # execute the pulse sequence
    options = dict(
        nshots=params.nshots,
        averaging_mode=AveragingMode.CYCLIC,
        acquisition_type=AcquisitionType.DISCRIMINATION,
    )
    if params.unrolling:
        results = platform.execute(sequences, **options)
    else:
        results = {}
        for sequence in sequences:
            results.update(platform.execute([sequence], **options))

    for gates, ro_pulses in zip(gatelist, all_ro_pulses):
        gate = "-".join(gates)
        for qubit in targets:
            prob = results[ro_pulses[qubit].id]
            z_proj = 1 - 2 * prob
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


def apply_drag(pulse: Pulse, beta_param: Optional[float] = None) -> Pulse:
    """Apply Drag with parameter beta."""
    if beta_param is None:
        return pulse
    return replace(  # pragma: no cover
        pulse,
        envelope=Drag(
            rel_sigma=pulse.envelope.rel_sigma,
            beta=beta_param,
        ),
    )


def allxy_sequence(
    platform: CalibrationPlatform,
    gates,
    qubit,
    sequence_delay=None,
    readout_delay=None,
    beta_param=None,
):
    natives = platform.natives.single_qubit[qubit]
    qd_channel, _ = natives.RX()[0]
    sequence = PulseSequence()
    if sequence_delay is not None:
        sequence.append((qd_channel, Delay(duration=sequence_delay)))
    for gate in gates:
        if gate == "I":
            pass

        if gate == "Xp":
            qd_channel, rx_pulse = natives.RX()[0]
            sequence.append((qd_channel, apply_drag(rx_pulse, beta_param)))

        if gate == "X9":
            rx90_sequence = natives.R(theta=np.pi / 2)
            for channel, pulse in rx90_sequence:
                sequence.append((channel, apply_drag(pulse, beta_param)))

        if gate == "Yp":
            ry_sequence = natives.R(phi=np.pi / 2)
            for channel, pulse in ry_sequence:
                sequence.append((channel, apply_drag(pulse, beta_param)))
        if gate == "Y9":
            ry90_sequence = natives.R(theta=np.pi / 2, phi=np.pi / 2)
            for channel, pulse in ry90_sequence:
                sequence.append((channel, apply_drag(pulse, beta_param)))

    # RO pulse starting just after pair of gates
    qd_channel = platform.qubits[qubit].drive
    ro_channel, ro_pulse = natives.MZ()[0]
    if readout_delay is not None:
        sequence.append(
            (
                ro_channel,
                Delay(duration=sequence.channel_duration(qd_channel) + readout_delay),
            )
        )
    else:
        sequence.append(
            (
                ro_channel,
                Delay(duration=sequence.channel_duration(qd_channel)),
            )
        )
    sequence.append((ro_channel, ro_pulse))
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
