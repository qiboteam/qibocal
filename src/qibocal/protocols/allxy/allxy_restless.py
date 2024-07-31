from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, Parameters, Results, Routine

from . import allxy


@dataclass
class AllXYResonatorParameters(Parameters):
    """AllXY Restless runcard inputs."""

    delay_start: float
    """Initial delay parameter for resonator depletion."""
    delay_end: float
    """Final delay parameter for resonator depletion."""
    delay_step: float
    """Step delay parameter for resonator depletion."""


@dataclass
class AllXYResonatorResults(Results):
    """AllXY outputs."""


AllXYRestlessType = np.dtype([("samples", np.int32), ("initial_state", np.int32)])


@dataclass
class AllXYRestlessData(Data):
    """AllXY acquisition outputs."""

    data: dict[tuple[QubitId, float], npt.NDArray[AllXYRestlessType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""

    @property
    def delay_params(self):
        """Access qubits from data structure."""
        return np.unique([b[1] for b in self.data])


def _acquisition(
    params: AllXYResonatorParameters,
    platform: Platform,
    targets: list[QubitId],
) -> AllXYRestlessData:
    r"""
    Data acquisition for allXY experiment varying delay after a measurement pulse to characterise resonator depletion time: https://arxiv.org/pdf/1604.00916.
    Passive resonator depletion time: Time it takes the process by which photons inside a resonator dissipate over time without any external intervention.
    After a measurement is performed, photons remain in the resonator and qubits errors induced if trying to drive the qubit by leftover photons due to the coupling
    resonator-qubit inducing a shift in the qubit frequency. This experiment is used to characterise the resonator depletion time by waiting an increased delay time
    until the allXY pattern looks right.
    """

    data = AllXYRestlessData()

    delays = np.arange(params.delay_start, params.delay_end, params.delay_step)
    # sweep the parameters
    for delay_param in delays:
        for gates in allxy.gatelist:
            # create a sequence of pulses
            ro_pulses = {}
            sequence = PulseSequence()
            for qubit in targets:
                ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
                sequence.add(ro_pulse)
                sequence, ro_pulses[qubit] = allxy.add_gate_pair_pulses_to_sequence(
                    platform,
                    gates,
                    qubit,
                    sequence,
                    sequence_delay=int(
                        delay_param
                    ),  # We need conversion to int due to devices for now
                    readout_delay=0,
                )

            # execute the pulse sequence
            results = platform.execute_pulse_sequence(
                sequence,
                ExecutionParameters(
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    averaging_mode=AveragingMode.SINGLESHOT,
                ),
            )

            # retrieve the results for every qubit
            for qubit in targets:
                samples = results[ro_pulses[qubit].serial].samples
                initial_state = results[ro_pulse.serial].samples
                # store the results
                gate = "-".join(gates)
                data.register_qubit(
                    AllXYRestlessType,
                    (qubit, float(delay_param), gate),
                    dict(samples=samples, initial_state=initial_state),
                )
    return data


def _fit(_data: AllXYRestlessData) -> AllXYResonatorResults:
    """Post-processing for allXYDrag."""
    return AllXYResonatorResults()


def _plot(data: AllXYRestlessData, target: QubitId, fit: AllXYResonatorResults = None):
    """Plotting function for allXYDrag."""

    figures = []
    fitting_report = ""
    fig = go.Figure()

    delay_params = data.delay_params
    probs = defaultdict(list)
    gates = ["-".join(gate) for gate in allxy.gatelist]
    for j, delay_param in enumerate(delay_params):
        for gate in gates:
            # import pdb; pdb.set_trace()
            delay_param_samples = data[target, delay_param, gate].samples
            initial_states = data[target, delay_param, gate].initial_state
            processed_samples = delay_param_samples
            # TODO: Fancier logic to invert the samples by multiplying
            for i, initial_state in enumerate(initial_states):
                if initial_state == 1:
                    processed_samples[i] = 1 - delay_param_samples[i]
            # import pdb; pdb.set_trace()
            probs[delay_param].append(2 * np.mean(1 - processed_samples) - 1)

        fig.add_trace(
            go.Scatter(
                x=gates,
                y=probs[delay_param],
                mode="markers+lines",
                opacity=0.5,
                name=f"Delay {delay_param}",
                showlegend=True,
                legendgroup=f"group{j}",
                text=allxy.gatelist,
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


allxy_restless = Routine(_acquisition, _fit, _plot)
"""AllXYDrag Routine object."""
