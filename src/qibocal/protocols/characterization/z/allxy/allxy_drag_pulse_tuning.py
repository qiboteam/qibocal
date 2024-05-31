from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

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
class AllXYDragParameters(Parameters):
    """AllXYDrag runcard inputs."""

    beta_start: float
    """Initial beta parameter for Drag pulse."""
    beta_end: float
    """Final beta parameter for Drag pulse."""
    beta_step: float
    """Step beta parameter for Drag pulse."""
    unrolling: bool = False
    """If ``True`` it uses sequence unrolling to deploy multiple sequences in a single instrument call.
    Defaults to ``False``."""


@dataclass
class AllXYDragResults(Results):
    """AllXYDrag outputs."""


@dataclass
class AllXYDragData(Data):
    """AllXY acquisition outputs."""

    beta_param: Optional[float] = None
    """Beta parameter for drag pulse."""
    data: Dict[Tuple[QubitId, float], npt.NDArray[allxy.AllXYType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""

    @property
    def beta_params(self):
        """Access qubits from data structure."""
        keys = list(self.data.keys())
        print("Contenido de las claves de self.data:", keys)
        for key in keys:
            print(f"Clave: {key}, tipo: {type(key)}")
        return np.unique(
            [
                key[1]
                for key in self.data.keys()
                if isinstance(key, tuple) and len(key) > 1
            ]
        )

    def register_qubit(self, data_type, qubit, data):
        key = (qubit, self.beta_param)
        if key not in self.data:
            self.data[key] = np.array(
                [(data["prob"][0], data["gate"][0], data["errors"][0])], dtype=data_type
            )
        else:
            self.data[key] = np.append(
                self.data[key],
                np.array(
                    [(data["prob"][0], data["gate"][0], data["errors"][0])],
                    dtype=data_type,
                ),
            )


def _acquisition(
    params: AllXYDragParameters,
    platform: Platform,
    qubits: list[QubitId],
) -> AllXYDragData:
    r"""
    Data acquisition for allXY experiment varying beta.
    The AllXY experiment is a simple test of the calibration of single qubit gatesThe qubit (initialized in the |0> state)
    is subjected to two back-to-back single-qubit gates and measured. In each round, we run 21 different gate pairs:
    ideally, the first 5 return the qubit to |0>, the next 12 drive it to superposition state, and the last 4 put the
    qubit in |1> state.

    The AllXY iteration method allows the user to execute iteratively the list of gates playing with the drag pulse shape
    in order to find the optimal drag pulse coefficient for pi pulses.
    """

    data = AllXYDragData()

    betas = np.arange(params.beta_start, params.beta_end, params.beta_step).round(4)
    # sweep the parameters
    # repeat the experiment as many times as defined by software_averages
    # for iteration in range(params.software_averages):
    sequences, all_ro_pulses = [], []
    for beta_param in betas:
        print(beta_param)
        data.beta_param = (
            beta_param  # Asigna el valor actual de beta_param a data.beta_param
        )
        for gates in allxy.gatelist:
            sequences.append(PulseSequence())
            all_ro_pulses.append({})
            for qubit in qubits:
                (
                    sequences[-1],
                    all_ro_pulses[-1][qubit],
                ) = allxy.add_gate_pair_pulses_to_sequence(
                    platform, gates, qubit, sequences[-1], beta_param=beta_param
                )

        # execute the pulse sequence
        options = ExecutionParameters(
            nshots=params.nshots,
            averaging_mode=AveragingMode.CYCLIC,
            relaxation_time=params.relaxation_time,
        )
        if params.unrolling:
            results = platform.execute_pulse_sequences(sequences, options)
        else:
            results = [
                platform.execute_pulse_sequence(sequence, options)
                for sequence in sequences
            ]

        for ig, (gates, ro_pulses) in enumerate(zip(allxy.gatelist, all_ro_pulses)):
            gate = "-".join(gates)
            for qubit in qubits:
                serial = ro_pulses[qubit].serial
                if params.unrolling:
                    prob = results[serial][ig].probability(state=1)
                    z_proj = 1 - 2 * prob
                else:
                    prob = results[ig][serial].probability(state=1)
                    # z_proj = prob #2 * prob - 1
                    # prob = results[serial][ig].probability(1)
                    z_proj = 1 - 2 * prob

                errors = 2 * np.sqrt(prob * (1 - prob) / params.nshots)
                data.register_qubit(
                    allxy.AllXYType,
                    qubit,
                    dict(
                        prob=np.array([z_proj]),
                        gate=np.array([gate]),
                        errors=np.array([errors]),
                    ),
                )

        # finalmente, guarda los datos restantes
    return data


def _fit(_data: AllXYDragData) -> AllXYDragResults:
    """Post-processing for allXYDrag."""
    return AllXYDragResults()


def _plot(data: AllXYDragData, qubit: QubitId, fit: AllXYDragResults = None):
    """Plotting function for allXYDrag."""

    figures = []
    fitting_report = ""

    fig = go.Figure()
    beta_params = data.beta_params

    for j, beta_param in enumerate(beta_params):
        beta_param_data = data[(qubit, beta_param)]
        fig.add_trace(
            go.Scatter(
                x=beta_param_data["gate"],
                y=beta_param_data["prob"],
                mode="markers+lines",
                opacity=0.5,
                name=f"Beta {beta_param}",
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


allxy_drag_pulse_tuning = Routine(_acquisition, _fit, _plot)
"""AllXYDrag Routine object."""
