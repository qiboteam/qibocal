import json
import math
from collections import defaultdict

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibo import gates
from qibo.config import log

# from qibolab import Platform
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.transpilers.unitary_decompositions import u3_decomposition

from qibocal.calibrations.niGSC.basics.fitting import exp1B_func, fit_exp1B_func
from qibocal.calibrations.niGSC.basics.utils import gate_fidelity
from qibocal.config import log
from qibocal.plots.utils import get_color


def RB_fit(x, A, p, B):
    """A*p^x+B fit"""
    return A * p**x + B


def plot(data, fit, qubit):
    if data.__class__.__name__ == "StdRBData":
        quantity = "length"
        unit = "dimensionless"
        title = "Sequence length"
        fitting = RB_fit

    figures = []
    fitting_report = ""

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=("Standard RB",),
    )

    qubit_data = data.df[data.df["qubit"] == qubit]
    RB_parameters = qubit_data[quantity].pint.to(unit).pint.magnitude.unique()

    fig.add_trace(
        go.Scatter(
            x=qubit_data[quantity].pint.to(unit).pint.magnitude,
            y=qubit_data["probabilities"].pint.to("dimensionless").pint.magnitude,
            marker_color=get_color(0),
            mode="markers",
            opacity=0.3,
            name="StandardRB",
            showlegend=True,
            legendgroup="StandardRB",
        ),
        row=1,
        col=1,
    )

    # add fitting trace
    if len(data) > 0:
        RB_parameter_range = np.linspace(
            min(RB_parameters),
            max(RB_parameters),
            2 * len(data),
        )
        params = fit.fitted_parameters[qubit]

        fig.add_trace(
            go.Scatter(
                x=RB_parameter_range,
                y=fitting(RB_parameter_range, *params),
                name=f"A: {params[0]:.3f}, p: {params[1]:.3f}, B: {params[2]:.3f}",
                line=go.scatter.Line(dash="dot"),
                marker_color="rgb(255, 130, 67)",
            ),
            row=1,
            col=1,
        )

        fitting_report += f"{qubit} | p: {float(params[1]):.3f}<br>"

        fitting_report += f"{qubit} | fidelity primitive: {float(fit.fidelities_primitive[qubit]):.3f}<br>"

        fitting_report += f"{qubit} | fidelity: {float(fit.fidelities[qubit]):.3f}<br>"

        fitting_report += f"{qubit} | Average error per gate(%): {float(fit.average_errors_gate[qubit]):.3f}<br>"

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title=title,
        yaxis_title="Survival probability",
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
    4: lambda q: gates.X(q),  # gates.U3(q, np.pi, 0, np.pi),
    5: lambda q: gates.Y(q),  # U3(q, np.pi, 0, 0),
    # pi/2 rotations
    6: lambda q: gates.RX(q, np.pi / 2),  # U3(q, np.pi / 2, -np.pi / 2, np.pi / 2),
    7: lambda q: gates.RX(q, -np.pi / 2),  # U3(q, -np.pi / 2, -np.pi / 2, np.pi / 2),
    8: lambda q: gates.RY(q, np.pi / 2),  # U3(q, np.pi / 2, 0, 0),
    9: lambda q: gates.RY(q, -np.pi / 2),  # U3(q, -np.pi / 2, 0, 0),
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
        circuits = defaultdict(list)
        for depth in self.depths:
            size = 1000 // (2 * depth)
            unrolling_runs = math.ceil(self.runs / size)
            print(size, unrolling_runs)
            for run in range(unrolling_runs):
                circuit = list(np.random.randint(0, len(INT_TO_GATE), depth))

                sequences[f"{depth}_{run}"].append(
                    self.circuit_to_sequence(self.platform, qubit, circuit, size=size)
                )
                circuits[f"{depth}_{run}"].append(circuit)

        return sequences, circuits

    def inverse(self, ints, q=0):
        unitary = np.linalg.multi_dot([INT_TO_GATE[i](q).matrix for i in ints[::-1]])
        inverse_unitary = np.transpose(np.conj(unitary))
        theta, phi, lam = u3_decomposition(inverse_unitary)
        return gates.U3(q, theta, phi, lam)

    def circuit_to_sequence(self, platform: Platform, qubit, circuit, size):
        # Define PulseSequence
        sequence = PulseSequence()
        virtual_z_phases = defaultdict(int)
        next_pulse_start = 0
        for i in range(size):
            # virtual_z_phases = defaultdict(int)
            gate_number = 0
            for index in circuit:
                if index == 0:
                    continue
                gate = INT_TO_GATE[index](qubit)
                # Virtual gates
                if isinstance(gate, gates.Z):
                    virtual_z_phases[qubit] += np.pi
                if isinstance(gate, gates.RZ):
                    virtual_z_phases[qubit] += gate.parameters[0]
                # X
                if isinstance(gate, (gates.X, gates.Y)):
                    phase = 0 if isinstance(gate, gates.X) else -np.pi / 2
                    sequence.add(
                        platform.create_RX_pulse(
                            qubit,
                            start=next_pulse_start,
                            relative_phase=virtual_z_phases[qubit] + phase,
                        )
                    )
                # RX
                if isinstance(gate, (gates.RX, gates.RY)):
                    phase = 0 if isinstance(gate, gates.RX) else -np.pi / 2
                    phase += 0 if gate.parameters[0] > 0 else -np.pi
                    sequence.add(
                        platform.create_RX90_pulse(
                            qubit,
                            start=next_pulse_start,
                            relative_phase=virtual_z_phases[qubit] + phase,
                        )
                    )
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

                if gate_number == 0:
                    if isinstance(gate, gates.Z):
                        pass
                    elif isinstance(gate, gates.RZ):
                        pass
                    else:
                        next_pulse_start = sequence.finish

                gate_number += 1

            invert_gate = self.inverse(circuit, qubit)
            # U3 pulses
            if isinstance(invert_gate, gates.U3):
                theta, phi, lam = invert_gate.parameters
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

            # Add measurement pulse
            measurement_start = sequence.finish

            MZ_pulse = platform.create_MZ_pulse(qubit, start=measurement_start)
            sequence.add(MZ_pulse)
            next_pulse_start = sequence.finish
            next_pulse_start += platform.relaxation_time

        return sequence
