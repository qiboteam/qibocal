from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AveragingMode, PulseSequence

from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform

from .allxy import AllXYType, allxy_sequence, gatelist


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


@dataclass
class AllXYResonatorResults(Results):
    """AllXYDrag outputs."""


@dataclass
class AllXYResonatorData(Data):
    """AllXY acquisition outputs."""

    delay_param: Optional[float] = None
    """Delay parameter for resonator depletion."""
    data: dict[tuple[QubitId, float], npt.NDArray[AllXYType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""

    @property
    def delay_params(self):
        """Access qubits from data structure."""
        return np.unique([b[1] for b in self.data])


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

    data = AllXYResonatorData()

    delays = np.arange(params.delay_start, params.delay_end, params.delay_step)
    # sweep the parameters
    for delay in delays:
        sequences, all_ro_pulses = [], []
        for gates in gatelist:
            sequence = PulseSequence()
            ro_pulses = {}
            for qubit in targets:
                qubit_sequence, ro_pulses[qubit] = allxy_sequence(
                    platform,
                    gates,
                    qubit,
                    beta_param=params.beta_param,
                    sequence_delay=delay,
                    readout_delay=1000,
                )
                sequence += qubit_sequence
            sequences.append(sequence)
            all_ro_pulses.append(ro_pulses)
        options = dict(nshots=params.nshots, averaging_mode=AveragingMode.CYCLIC)
        if params.unrolling:
            results = platform.execute(sequences, **options)
        else:
            results = {}
            for sequence in sequences:
                results.update(platform.execute([sequence], **options))

        for gates, ro_pulses in zip(gatelist, all_ro_pulses):
            gate = "-".join(gates)
            for qubit in targets:
                prob = 1 - results[ro_pulses[qubit].id]
                z_proj = 2 * prob - 1
                errors = 2 * np.sqrt(prob * (1 - prob) / params.nshots)
                data.register_qubit(
                    AllXYType,
                    (qubit, float(delay)),
                    dict(
                        prob=np.array([z_proj]),
                        gate=np.array([gate]),
                        errors=np.array([errors]),
                    ),
                )

    return data


def _fit(_data: AllXYResonatorData) -> AllXYResonatorResults:
    """Post-processing for allXYDrag."""
    return AllXYResonatorResults()


def _plot(data: AllXYResonatorData, target: QubitId, fit: AllXYResonatorResults = None):
    """Plotting function for allXYDrag."""

    figures = []
    fitting_report = ""

    fig = go.Figure()
    delay_params = data.delay_params

    for j, delay_param in enumerate(delay_params):
        delay_param_data = data[target, delay_param]
        fig.add_trace(
            go.Scatter(
                x=delay_param_data.gate,
                y=delay_param_data.prob,
                mode="markers+lines",
                opacity=0.5,
                name=f"Delay {delay_param}",
                showlegend=True,
                legendgroup=f"group{j}",
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
