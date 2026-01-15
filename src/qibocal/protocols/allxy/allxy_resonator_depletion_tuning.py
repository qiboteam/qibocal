from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
from qibolab import AveragingMode, Parameter, PulseSequence, Sweeper

from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform

from .allxy import allxy_sequence, gatelist


@dataclass
class AllXYResonatorParameters(Parameters):
    """AllXYDrag runcard inputs."""

    delay_start: float
    """Initial delay parameter for resonator depletion."""
    delay_end: float
    """Final delay parameter for resonator depletion."""
    delay_step: float
    """Step delay parameter for resonator depletion."""
    readout_delay: int = 1000
    """Delay on readout."""
    unrolling: bool = False
    """If ``True`` it uses sequence unrolling to deploy multiple sequences in a single instrument call.
    Defaults to ``False``."""
    beta_param: float = None
    """Beta parameter for drag pulse."""

    @property
    def delay_range(self) -> np.ndarray:
        return np.arange(self.delay_start, self.delay_end, self.delay_step)


@dataclass
class AllXYResonatorResults(Results):
    """AllXYDrag outputs."""


@dataclass
class AllXYResonatorData(Data):
    """AllXY acquisition outputs."""

    delays: list[float] = field(default_factory=dict)
    data: dict[QubitId, np.ndarray] = field(default_factory=dict)
    """Raw data acquired."""

    def z(self, qubit: QubitId) -> np.ndarray:
        return 2 * self.data[qubit] - 1


def _acquisition(
    params: AllXYResonatorParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> AllXYResonatorData:
    r"""
    Data acquisition for allXY experiment varying delay after a measurement pulse to characterise resonator depletion time: https://arxiv.org/pdf/1604.00916.
    Passive resonator depletion time: Time it takes the process by which photons inside a resonator dissipate over time without any external intervention.
    After a measurement is performed, photons remain in the resonator and qubits errors induced if trying to drive the qubit by leftover photons due to the coupling
    resonator-qubit inducing a shift in the qubit frequency. This experiment is used to characterise the resonator depletion time by waiting an increased delay time
    until the allXY pattern looks right.
    """

    data = AllXYResonatorData(delays=params.delay_range.tolist())
    data_ = {qubit: [] for qubit in targets}

    for gates in gatelist:
        sequence = PulseSequence()
        ro_pulses = {}
        all_delays = []
        for qubit in targets:
            qubit_sequence, delays, ro_pulses[qubit] = allxy_sequence(
                platform,
                gates,
                qubit,
                beta_param=params.beta_param,
                readout_delay=1000,
            )
            all_delays += delays
            sequence += qubit_sequence

        sweeper = Sweeper(
            parameter=Parameter.duration,
            values=params.delay_range,
            pulses=all_delays,
        )

        results = platform.execute(
            [sequence],
            [[sweeper]],
            nshots=params.nshots,
            averaging_mode=AveragingMode.CYCLIC,
        )

        for qubit in targets:
            prob = 1 - results[ro_pulses[qubit].id]
            data_[qubit].append(prob)

    data.data = {qubit: np.column_stack(data_[qubit]) for qubit in targets}

    return data


def _fit(_data: AllXYResonatorData) -> AllXYResonatorResults:
    """Post-processing for allXYDrag."""
    return AllXYResonatorResults()


def _plot(data: AllXYResonatorData, target: QubitId, fit: AllXYResonatorResults = None):
    """Plotting function for allXYDrag."""

    figures = []
    fitting_report = ""

    fig = go.Figure()
    for i, data_ in enumerate(data.z(target)):
        fig.add_trace(
            go.Scatter(
                x=["".join(gate) for gate in gatelist],
                y=data_,
                mode="markers+lines",
                opacity=0.5,
                name=f"Delay {data.delays[i]}",
                showlegend=True,
                legendgroup=f"group{i}",
                text=gatelist,
                textposition="bottom center",
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


allxy_resonator_depletion_tuning = Routine(_acquisition, _fit, _plot)
"""AllXYDrag Routine object."""
