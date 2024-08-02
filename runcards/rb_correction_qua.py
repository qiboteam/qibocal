from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import create_platform
from qibolab.qubits import QubitId

from qibocal.auto.execute import Executor
from qibocal.auto.operation import Data
from qibocal.cli.report import report
from qibocal.protocols.randomized_benchmarking import utils

RBCorrectionType = np.dtype(
    [
        ("biases", np.float64),
        ("qubit_frequency", np.float64),
        ("pulse_fidelity_uncorrected", np.float64),
        ("pulse_fidelity_corrected", np.float64),
    ]
)
"""Custom dtype for RBCorrection routines."""


biases = np.arange(-0.2, 0.1, 0.01)
"bias points to sweep"

# Flipping
nflips_max = 200
"""Maximum number of flips ([RX(pi) - RX(pi)] sequences). """
nflips_step = 10
"""Flip step."""

#  Ramsey signal
detuning = 3_000_000
"""Frequency detuning [Hz]."""
delay_between_pulses_start = 16
"""Initial delay between RX(pi/2) pulses in ns."""
delay_between_pulses_end = 5_000
"""Final delay between RX(pi/2) pulses in ns."""
delay_between_pulses_step = 200
"""Step delay between RX(pi/2) pulses in ns."""

# Std Rb Ondevice Parameters
num_of_sequences: int = 100
max_circuit_depth: int = 250
"Maximum circuit depth"
delta_clifford: int = 50
"Play each sequence with a depth step equals to delta_clifford"
seed: Optional[int] = 1234
"Pseudo-random number generator seed"
n_avg: int = 128
"Number of averaging loops for each random sequence"
save_sequences: bool = True
apply_inverse: bool = True
state_discrimination: bool = True
"Flag to enable state discrimination if the readout has been calibrated (rotated blobs and threshold)"


@dataclass
class RBCorrectionSignalData(Data):
    """Coherence acquisition outputs."""

    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""

    @property
    def average(self):
        if len(next(iter(self.data.values())).shape) > 1:
            return utils.average_single_shots(self.__class__, self.data)
        return self


parser = ArgumentParser()
parser.add_argument("--target", default="D2", help="Target qubit index")
parser.add_argument("--platform", type=str, default="qw11q", help="Platform name")
parser.add_argument(
    "--path", type=str, default="TESTRBCorrection", help="Path for the output"
)
args = parser.parse_args()

target = args.target
path = args.path

data = RBCorrectionSignalData()

platform = create_platform(args.platform)
for target in [args.target]:

    # NOTE: Center around the sweetspot [Optional]
    centered_biases = biases + platform.qubits[target].sweetspot

    for i, bias in enumerate(centered_biases):
        with Executor.open(
            f"myexec_{i}",
            path=path / Path(f"flux_{i}"),
            platform=platform,
            targets=[target],
            update=True,
            force=True,
        ) as e:

            rb_output_uncorrected = e.rb_ondevice(
                num_of_sequences=num_of_sequences,
                max_circuit_depth=max_circuit_depth,
                delta_clifford=delta_clifford,
                seed=seed,
                n_avg=n_avg,
                save_sequences=save_sequences,
                apply_inverse=apply_inverse,
                state_discrimination=state_discrimination,
            )

            ramsey_output = e.ramsey_signal(
                delay_between_pulses_start=delay_between_pulses_start,
                delay_between_pulses_end=delay_between_pulses_end,
                delay_between_pulses_step=delay_between_pulses_step,
                detuning=detuning,
            )
            flipping_output = e.flipping_signal(
                nflips_max=nflips_max,
                nflips_step=nflips_step,
            )

            discrimination_output = e.single_shot_classification(nshots=5000)

            rb_output_corrected = e.rb_ondevice(
                num_of_sequences=num_of_sequences,
                max_circuit_depth=max_circuit_depth,
                delta_clifford=delta_clifford,
                seed=seed,
                n_avg=n_avg,
                save_sequences=save_sequences,
                apply_inverse=apply_inverse,
                state_discrimination=state_discrimination,
            )

            data.register_qubit(
                RBCorrectionType,
                (target),
                dict(
                    biases=[bias],
                    qubit_frequency=[platform.qubits[target].native_gates.RX.frequency],
                    pulse_fidelity_uncorrected=[
                        rb_output_uncorrected.results.pars[target][
                            2
                        ]  # Get error covs[2]
                    ],
                    pulse_fidelity_corrected=[
                        rb_output_corrected.results.pars[target][2]
                    ],
                ),
            )

        report(e.path, e.history)


def plot(data: RBCorrectionSignalData, target: QubitId, path):
    """Plotting function for Coherence experiment."""

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=data[target].qubit_frequency,
            y=data[target].pulse_fidelity_uncorrected,
            opacity=1,
            name="Pulse Fidelity Uncorrected",
            showlegend=True,
            legendgroup="Pulse Fidelity Uncorrected",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=data[target].qubit_frequency,
            y=data[target].pulse_fidelity_corrected,
            opacity=1,
            name="Pulse Fidelity Corrected",
            showlegend=True,
            legendgroup="Pulse Fidelity Corrected",
        )
    )

    # last part
    figure.update_layout(
        showlegend=True,
        xaxis_title="Frequency [GHZ]",
        yaxis_title="Pulse Fidelity",
    )

    if path is not None:
        figure.write_html(path / Path("plot.html"))


plot(data, target, path=path)
