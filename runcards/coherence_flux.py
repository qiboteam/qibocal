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
from qibocal.auto.history import History
from qibocal.auto.operation import Data
from qibocal.cli.report import report
from qibocal.protocols.flux_dependence.utils import transmon_frequency

CoherenceFluxType = np.dtype(
    [
        ("biases", np.float64),
        ("qubit_frequency", np.float64),
        ("T1", np.float64),
        ("T2", np.float64),
        ("T2_ramsey", np.float64),
    ]
)
"""Custom dtype for CoherenceFlux routines."""


biases = np.arange(-0.2, 0.1, 0.01)
"bias points to sweep"

# Qubit spectroscopy
freq_width = 50_000_000
"""Width [Hz] for frequency sweep relative  to the qubit frequency."""
freq_step = 1_000_000
"""Frequency [Hz] step for sweep."""
drive_duration = 1000
"""Drive pulse duration [ns]. Same for all qubits."""

# Rabi amp signal
min_amp_factor = 0.0
"""Minimum amplitude multiplicative factor."""
max_amp_factor = 0.5
"""Maximum amplitude multiplicative factor."""
step_amp_factor = 0.01
"""Step amplitude multiplicative factor."""

# Flipping
nflips_max = 200
"""Maximum number of flips ([RX(pi) - RX(pi)] sequences). """
nflips_step = 10
"""Flip step."""

# T1 signal
delay_before_readout_start = 16
"""Initial delay before readout [ns]."""
delay_before_readout_end = 100_000
"""Final delay before readout [ns]."""
delay_before_readout_step = 4_000
"""Step delay before readout [ns]."""

#  Ramsey signal
detuning = 3_000_000
"""Frequency detuning [Hz]."""
delay_between_pulses_start = 16
"""Initial delay between RX(pi/2) pulses in ns."""
delay_between_pulses_end = 5_000
"""Final delay between RX(pi/2) pulses in ns."""
delay_between_pulses_step = 200
"""Step delay between RX(pi/2) pulses in ns."""

# T2 and Ramsey signal
delay_between_pulses_start_T2 = 16
"""Initial delay between RX(pi/2) pulses in ns."""
delay_between_pulses_end_T2 = 25_000
"""Final delay between RX(pi/2) pulses in ns."""
delay_between_pulses_step_T2 = 500
"""Step delay between RX(pi/2) pulses in ns."""
single_shot_T2: bool = False
"""If ``True`` save single shot signal data."""

# Optional qubit spectroscopy
drive_amplitude: Optional[float] = None
"""Drive pulse amplitude (optional). Same for all qubits."""
hardware_average: bool = True
"""By default hardware average will be performed."""
# Optional rabi amp signal
pulse_length: Optional[float] = 40
"""RX pulse duration [ns]."""
# Optional T1 signal
single_shot_T1: bool = False
"""If ``True`` save single shot signal data."""


@dataclass
class CoherenceFluxSignalData(Data):
    """Coherence acquisition outputs."""

    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""


parser = ArgumentParser()
parser.add_argument("--target", type=int, default=0, help="Target qubit index")
parser.add_argument("--platform", type=str, default="dummy", help="Platform name")
parser.add_argument(
    "--path", type=str, default="TTESTCoherenceFlux", help="Path for the output"
)
args = parser.parse_args()

target = args.target
path = args.path

data = CoherenceFluxSignalData()

for target in [args.target]:

    platform = create_platform(args.platform)

    params_qubit = {
        "w_max": platform.qubits[
            target
        ].drive_frequency,  # FIXME: this is not the qubit frequency
        "xj": 0,
        "d": platform.qubits[target].asymmetry,
        "normalization": platform.qubits[target].crosstalk_matrix[target],
        "offset": -platform.qubits[target].sweetspot
        * platform.qubits[target].crosstalk_matrix[
            target
        ],  # Check is this the right one ???
        "crosstalk_element": 1,
        "charging_energy": platform.qubits[target].Ec,
    }

    fit_function = transmon_frequency

    # TODO: Center around the sweetspot ???
    biases += platform.qubits[target].sweetspot

    i = 0
    for bias in biases:
        i += 1

        with Executor.open(
            f"myexec_{i}",
            path=args.path / Path(f"flux_{i}"),
            platform=args.platform,
            targets=[target],
            update=True,
            force=True,
        ) as e:

            # Change the flux
            e.platform.qubits[target].flux.offset = bias

            # Change the qubit frequency
            qubit_frequency = fit_function(bias, **params_qubit)  # * 1e9
            e.platform.qubits[target].drive_frequency = qubit_frequency
            e.platform.qubits[target].native_gates.RX.frequency = qubit_frequency

            qubit_spectroscopy_output = e.qubit_spectroscopy(
                freq_width=freq_width,
                freq_step=freq_step,
                drive_duration=drive_duration,
                drive_amplitude=drive_amplitude,
                relaxation_time=5000,
                nshots=1024,
            )

            # Set maximun drive amplitude
            e.platform.qubits[target].native_gates.RX.amplitude = (
                0.5  # FIXME: For QM this should be 0.5
            )
            e.platform.qubits[target].native_gates.RX.duration = pulse_length

            rabi_output = e.rabi_amplitude_signal(
                min_amp_factor=min_amp_factor,
                max_amp_factor=max_amp_factor,
                step_amp_factor=step_amp_factor,
                pulse_length=e.platform.qubits[target].native_gates.RX.duration,
            )

            if rabi_output.results.amplitude[target] > 0.5:
                print(
                    f"Rabi fit has pi pulse amplitude {rabi_output.results.amplitude[target]}, greater than 0.5 not possible for QM. Skipping to next bias point."
                )
                e.platform.qubits[target].native_gates.RX.amplitude = (
                    0.5  # FIXME: For QM this should be 0.5
                )
                continue

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

            t1_output = e.t1_signal(
                delay_before_readout_start=delay_before_readout_start,
                delay_before_readout_end=delay_before_readout_end,
                delay_before_readout_step=delay_before_readout_step,
                single_shot=single_shot_T1,
            )

            # TODO: Estimate T2 with Ramsey signal without detuning
            t2_output = e.t2_signal(
                delay_between_pulses_start=delay_between_pulses_start_T2,
                delay_between_pulses_end=delay_between_pulses_end_T2,
                delay_between_pulses_step=delay_between_pulses_step_T2,
                single_shot=single_shot_T2,
            )

            ramsey_t2_output = e.ramsey_signal(
                delay_between_pulses_start=delay_between_pulses_start_T2,
                delay_between_pulses_end=delay_between_pulses_end_T2,
                delay_between_pulses_step=delay_between_pulses_step_T2,
                detuning=0,
            )

            data.register_qubit(
                CoherenceFluxType,
                (target),
                dict(
                    biases=[bias],
                    qubit_frequency=[
                        e.platform.qubits[target].native_gates.RX.frequency
                    ],
                    T1=[t1_output.results.t1[target][0]],
                    T2=[t2_output.results.t2[target][0]],
                    T2_ramsey=[ramsey_t2_output.results.t2[target][1]],
                ),
            )

            report(e.path, e.history)
            e.history = History()


def plot(data: CoherenceFluxSignalData, target: QubitId, path=None):
    """Plotting function for Coherence experiment."""

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=data[target].qubit_frequency,
            y=data[target].T1,
            opacity=1,
            name="T1",
            showlegend=True,
            legendgroup="T1",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=data[target].qubit_frequency,
            y=data[target].T2,
            opacity=1,
            name="T2",
            showlegend=True,
            legendgroup="T2",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=data[target].qubit_frequency,
            y=data[target].T2_ramsey,
            opacity=1,
            name="T2_ramsey",
            showlegend=True,
            legendgroup="T2_ramsey",
        )
    )

    figure.update_layout(
        showlegend=True,
        xaxis_title="Frequency [GHZ]",
        yaxis_title="Coherence [ns]",
    )

    if path is not None:
        figure.write_html(path / Path("plot.html"))


plot(data, target, path=args.path)
