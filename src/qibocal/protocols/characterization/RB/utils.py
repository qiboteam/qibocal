import json
from collections import defaultdict

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibo import gates
from qibo.config import log
from qibolab import Platform
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal.calibrations.niGSC.basics.fitting import exp1B_func, fit_exp1B_func
from qibocal.calibrations.niGSC.basics.utils import gate_fidelity
from qibocal.config import log
from qibocal.plots.utils import get_color


def fit(x, A, B, p):
    """A*p^x+B fit"""
    return A * p**x + B


def plot(data, fit, qubit):
    if data.__class__.__name__ == "StdRBData":
        quantity = "amplitude"
        unit = "dimensionless"
        title = "Amplitude (dimensionless)"
        fitting = fit

    figures = []
    fitting_report = ""

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (V)",
            "phase (rad)",
        ),
    )

    runs_data = [
        data[d][r]["hardware_probabilities"]
        for d in data["depths"]
        for r in range(data["runs"])
    ]

    fig.add_trace(
        go.Scatter(
            x=data["depths"] * data["runs"],
            y=runs_data,
            line=dict(color="#6597aa"),
            mode="markers",
            marker={"opacity": 0.2, "symbol": "square"},
            name="runs",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["depths"],
            y=data["groundstate probabilities"],
            line=dict(color="#aa6464"),
            mode="markers",
            name="average",
        )
    )

    # add fitting trace
    if len(data) > 0:
        rabi_parameter_range = np.linspace(
            min(rabi_parameters),
            max(rabi_parameters),
            2 * len(data),
        )
        params = fit.fitted_parameters[qubit]
        fig.add_trace(
            go.Scatter(
                x=rabi_parameter_range,
                y=fit(rabi_parameter_range, *params),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
                marker_color="rgb(255, 130, 67)",
            ),
            row=1,
            col=1,
        )

        fitting_report += (
            f"{qubit} | pi_pulse_amplitude: {float(fit.amplitude[qubit]):.3f}<br>"
        )

    fig.update_layout(
        {
            "title": f"Gate fidelity: {data['Gate fidelity']}. Gate fidelity primitive: {data['Gate fidelity primitive']}."
        }
    )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title=title,
        yaxis_title="MSR (uV)",
        xaxis2_title=title,
        yaxis2_title="Phase (rad)",
    )

    figures.append(fig)

    return figures, fitting_report


INT_TO_GATE = {
    # Virtual gates
    0: lambda q: gates.I(q),
    1: lambda q: gates.Z(q),
    2: lambda q: gates.RZ(q, np.pi / 2),
    3: lambda q: gates.RZ(q, -np.pi / 2),
    # pi rotations
    4: lambda q: gates.X(q),
    5: lambda q: gates.Y(q),
    # pi/2 rotations
    6: lambda q: gates.RX(q, np.pi / 2),
    7: lambda q: gates.RX(q, -np.pi / 2),
    8: lambda q: gates.RY(q, np.pi / 2),
    9: lambda q: gates.RY(q, -np.pi / 2),
    # 2pi/3 rotations
    10: lambda q: gates.U3(q, np.pi / 2, -np.pi / 2, 0),  # Rx(pi/2)Ry(pi/2)
    11: lambda q: gates.U3(q, np.pi / 2, -np.pi / 2, np.pi),  # Rx(pi/2)Ry(-pi/2)
    12: lambda q: gates.U3(q, np.pi / 2, np.pi / 2, 0),  # Rx(-pi/2)Ry(pi/2)
    13: lambda q: gates.U3(q, np.pi / 2, np.pi / 2, -np.pi),  # Rx(-pi/2)Ry(-pi/2)
    14: lambda q: gates.U3(q, np.pi / 2, 0, np.pi / 2),  # Ry(pi/2)Rx(pi/2)
    15: lambda q: gates.U3(q, np.pi / 2, 0, -np.pi / 2),  # Ry(pi/2)Rx(-pi/2)
    16: lambda q: gates.U3(q, np.pi / 2, -np.pi, np.pi / 2),  # Ry(-pi/2)Rx(pi/2)
    17: lambda q: gates.U3(q, np.pi / 2, np.pi, -np.pi / 2),  # Ry(-pi/2)Rx(-pi/2)
    # Hadamard-like
    18: lambda q: gates.U3(q, np.pi / 2, -np.pi, 0),  # X Ry(pi/2)
    19: lambda q: gates.U3(q, np.pi / 2, 0, np.pi),  # X Ry(-pi/2)
    20: lambda q: gates.U3(q, np.pi / 2, np.pi / 2, np.pi / 2),  # Y Rx(pi/2)
    21: lambda q: gates.U3(q, np.pi / 2, -np.pi / 2, -np.pi / 2),  # Y Rx(pi/2)
    22: lambda q: gates.U3(q, np.pi, -np.pi / 4, np.pi / 4),  # Rx(pi/2)Ry(pi/2)Rx(pi/2)
    23: lambda q: gates.U3(
        q, np.pi, np.pi / 4, -np.pi / 4
    ),  # Rx(-pi/2)Ry(pi/2)Rx(-pi/2)
}


class RBSequence:
    def __init__(self, platform, depths, runs):
        self.platform = platform
        self.depths = depths
        self.runs = runs

    def get_sequences(self, qubit):
        sequences = defaultdict(list)
        for depth in self.depths:
            for run in range(self.runs):
                circuit = list(np.random.randint(0, len(INT_TO_GATE), depth))
                sequences[f"{depth}_{run}"].append(
                    self.circuit_to_sequence(self.platform, qubit, circuit)
                )
        return sequences

    def circuit_to_sequence(self, platform: AbstractPlatform, qubit, circuit):
        # Define PulseSequence
        sequence = PulseSequence()
        virtual_z_phases = defaultdict(int)

        next_pulse_start = 0
        for index in circuit:
            if index == 0:
                continue
            gate = INT_TO_GATE[index](qubit)
            # Virtual gates
            if isinstance(gate, gates.Z):
                virtual_z_phases[qubit] += np.pi
            if isinstance(gate, gates.RZ):
                virtual_z_phases[qubit] += gate.parameters[0]
            # U3 pulses
            if isinstance(gate, gates.U3):
                theta, phi, lam = gate.parameters
                virtual_z_phases[qubit] += lam
                sequence.add(
                    platform.create_RX90_pulse(
                        qubit,
                        start=next_pulse_start,
                        relative_phase=virtual_z_phases[qubit],
                    )
                )
                virtual_z_phases[qubit] += theta
                sequence.add(
                    platform.create_RX90_pulse(
                        qubit,
                        start=sequence.finish,
                        relative_phase=virtual_z_phases[qubit] - np.pi,
                    )
                )
                virtual_z_phases[qubit] += phi
            if isinstance(gate, (gates.X, gates.Y)):
                phase = 0 if isinstance(gate, gates.X) else np.pi / 2
                sequence.add(
                    platform.create_RX_pulse(
                        qubit,
                        start=next_pulse_start,
                        relative_phase=virtual_z_phases[qubit] + phase,
                    )
                )
            if isinstance(gate, (gates.RX, gates.RY)):
                phase = 0 if isinstance(gate, gates.RX) else np.pi / 2
                phase -= 0 if gate.parameters[0] > 0 else np.pi
                sequence.add(
                    platform.create_RX90_pulse(
                        qubit,
                        start=next_pulse_start,
                        relative_phase=virtual_z_phases[qubit] + phase,
                    )
                )

            next_pulse_start = sequence.finish

        # Add measurement pulse
        measurement_start = sequence.finish

        MZ_pulse = platform.create_MZ_pulse(qubit, start=measurement_start)
        sequence.add(MZ_pulse)

        return sequence
