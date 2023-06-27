from collections import defaultdict

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibo import gates

# from qibolab import Platform
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.transpilers.unitary_decompositions import u3_decomposition

from qibocal.protocols.characterization.randomized_benchmarking.fitting import (
    exp1B_func as rb_fit,
)
from qibocal.protocols.characterization.randomized_benchmarking.utils import (
    SINGLE_QUBIT_CLIFFORDS,
)

# def rb_fit(x, a, p, b):
#     """A*p^x+B fit"""
#     return a * p**x + b


def plot(data, fit, qubit):
    if data.__class__.__name__ == "StdRBData":
        quantity = "length"
        unit = "dimensionless"
        title = "Sequence length"
        fitting = rb_fit

    figures = []
    fitting_report = ""

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=("Standard RB",),
    )

    qubit_data = data[qubit]
    RB_parameters = getattr(qubit_data, quantity)

    fig.add_trace(
        go.Scatter(
            x=RB_parameters,
            y=qubit_data.probabilities,
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
    if len(RB_parameters) > 0:
        RB_parameter_range = np.linspace(
            min(RB_parameters),
            max(RB_parameters),
            2 * len(RB_parameters),
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


class RBSequence:
    """Creates the random sequences to execute RB"""

    def __init__(self, platform, depths, runs):
        self.platform = platform
        self.depths = depths
        self.runs = runs

    def get_sequences_list(self, qubit):
        """Get the sequences as a list of PulseSequence for each depth"""
        sequences = []
        circuits = []
        ro_pulses = {}
        for depth in self.depths:
            new_sequences = []
            for run in range(self.runs):
                circuit = list(np.random.randint(0, len(SINGLE_QUBIT_CLIFFORDS), depth))

                new_sequences.append(
                    self.circuit_to_sequence_list(self.platform, qubit, circuit)
                )

                circuits.append(circuit)

            ro_pulses_list = []
            for sequence in new_sequences:
                ro_pulses_list.append(sequence.ro_pulses[0])
            ro_pulses[depth] = ro_pulses_list

            sequences += new_sequences

        return sequences, circuits, ro_pulses

    def inverse(self, ints, q=0):
        unitary = np.linalg.multi_dot(
            [SINGLE_QUBIT_CLIFFORDS[i](q).matrix for i in ints[::-1]]
        )
        inverse_unitary = np.transpose(np.conj(unitary))
        theta, phi, lam = u3_decomposition(inverse_unitary)
        return gates.U3(q, theta, phi, lam)

    def circuit_to_sequence_list(
        self,
        platform: Platform,
        qubit,
        circuit,
    ):
        """Build the sequence according to the random numbers we get"""
        # Define PulseSequence
        sequence = PulseSequence()
        virtual_z_phases = defaultdict(int)
        next_pulse_start = 0
        # virtual_z_phases = defaultdict(int)
        for index in circuit:
            if index == 0:
                continue
            gate = SINGLE_QUBIT_CLIFFORDS[index](qubit)
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

        return sequence
