from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import DataUnits


@dataclass
class RoFrequencyParameters(Parameters):
    """RoFrequency runcard inputs."""

    freq_width: int
    """Width [Hz] for frequency sweep relative  to the qubit frequency."""
    freq_step: int
    """Frequency [Hz] step for sweep."""
    nshots: Optional[int]
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class RoFrequencyResults(Results):
    """RoFrequency outputs."""

    rotation_angle: Dict[Union[str, int], list]
    threshold: Dict[Union[str, int], list]
    fidelity: Dict[Union[str, int], list]
    assignment_fidelity: Dict[Union[str, int], list]
    average_state0: Dict[Union[str, int], list]
    average_state1: Dict[Union[str, int], list]


class RoFrequencyData(DataUnits):
    """RoFrequency acquisition outputs."""

    def __init__():
        super().__init__(
            name="data",
            quantities={"frequency": "Hz", "delta_frequency": "Hz"},
            options=["qubit", "iteration", "state"],
        )


def _acquisition(
    params: RoFrequencyParameters, platform: Platform, qubits: Qubits
) -> RoFrequencyData:
    """Data acquisition for readout optimization."""
    # create a sequence of pulses for the experiment:
    # long drive probing pulse - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    platform.reload_settings()

    # create two sequences of pulses:
    # state0_sequence: I  - MZ
    # state1_sequence: RX - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    state0_sequence = PulseSequence()
    state1_sequence = PulseSequence()

    RX_pulses = {}
    ro_pulses = {}
    for qubit in qubits:
        RX_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX_pulses[qubit].finish
        )

        state0_sequence.add(ro_pulses[qubit])
        state1_sequence.add(RX_pulses[qubit])
        state1_sequence.add(ro_pulses[qubit])
    sequences = {0: state0_sequence, 1: state1_sequence}
    # create a DataUnits object to store the results
    data = RoFrequencyData()
    # iterate over the frequency range
    delta_frequency_range = np.arange(
        -params.frequency_width / 2, params.frequency_width / 2, params.frequency_step
    )

    frequency_sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[ro_pulses[qubit] for qubit in qubits],
    )

    # Execute sequences for both states
    for state in [0, 1]:
        results = platform.sweep(
            sequences[state], frequency_sweeper, nshots=params.nshots, average=False
        )

        # retrieve and store the results for every qubit)
        for qubit in qubits:
            r = {k: v.ravel() for k, v in results[ro_pulses[qubit].serial].raw.items()}
            r.update(
                {
                    "frequency[Hz]": np.repeat(
                        np.vstack(delta_frequency_range).T,
                        params.nshots,
                        axis=0,
                    ).flatten()
                    + ro_pulses[qubit].frequency,
                    "delta_frequency[Hz]": np.repeat(
                        np.vstack(delta_frequency_range).T,
                        params.nshots,
                        axis=0,
                    ).flatten(),
                    "qubit": [qubit] * params.nshots * len(delta_frequency_range),
                    "iteration": np.repeat(
                        np.vstack(np.arange(params.nshots)).T,
                        len(delta_frequency_range),
                        axis=1,
                    ).flatten(),
                    "state": [state] * params.nshots * len(delta_frequency_range),
                }
            )
            data.add_data_from_dict(r)

    # finally, save the remaining data and the fits
    return data


def _fit(data: RoFrequencyData) -> RoFrequencyResults:
    """Post-processing function for RoFrequency."""
    labels = [
        "state",
        "qubit",
        "delta_frequency",
        "iteration",
    ]
    qubits = data.df["qubit"].unique()
    # data_fit = Data(
    #    name="fit",
    #    quantities=quantities,
    # )

    # Create a ndarray for i and q shots for all labels
    # shape=(i + j*q, qubit, state, label1, label2, ...)

    shape = (*[len(data.df[label].unique()) for label in labels],)
    nb_shots = len(data.df["iteration"].unique())

    iq_complex = data.df["i"].pint.magnitude.to_numpy().reshape(shape) + 1j * data.df[
        "q"
    ].pint.magnitude.to_numpy().reshape(shape)

    # Move state to 0, and iteration to -1
    iq_complex = np.moveaxis(
        iq_complex, [labels.index("state"), labels.index("iteration")], [0, -1]
    )

    # Take the mean ground state
    mean_gnd_state = np.mean(iq_complex[0, ...], axis=-1, keepdims=True)
    mean_exc_state = np.mean(iq_complex[1, ...], axis=-1, keepdims=True)
    angle = np.angle(mean_exc_state - mean_gnd_state)

    # Rotate the data
    iq_complex = iq_complex * np.exp(-1j * angle)

    # Take the cumulative distribution of the real part of the data
    iq_complex_sorted = np.sort(iq_complex.real, axis=-1)

    def cum_dist(complex_row):
        state0 = complex_row.real
        state1 = complex_row.imag
        combined = np.sort(np.concatenate((state0, state1)))

        # Compute the indices where elements in state0 and state1 would be inserted in combined
        idx_state0 = np.searchsorted(combined, state0, side="left")
        idx_state1 = np.searchsorted(combined, state1, side="left")

        # Create a combined histogram for state0 and state1
        hist_combined = np.bincount(
            idx_state0, minlength=len(combined)
        ) + 1j * np.bincount(idx_state1, minlength=len(combined))

        return hist_combined.cumsum()

    cum_dist = (
        np.apply_along_axis(
            func1d=cum_dist,
            axis=-1,
            arr=iq_complex_sorted[0, ...] + 1j * iq_complex_sorted[1, ...],
        )
        / nb_shots
    )

    # Find the threshold for which the difference between the cumulative distribution of the two states is maximum
    argmax = np.argmax(np.abs(cum_dist.real - cum_dist.imag), axis=-1, keepdims=True)

    # Use np.take_along_axis to get the correct indices for the threshold calculation
    threshold = np.take_along_axis(
        np.concatenate((iq_complex_sorted[0, ...], iq_complex_sorted[1, ...]), axis=-1),
        argmax,
        axis=-1,
    )

    # Calculate the fidelity
    fidelity = np.take_along_axis(
        np.abs(cum_dist.real - cum_dist.imag), argmax, axis=-1
    )
    assignment_fidelity = (
        1
        - (
            1
            - np.take_along_axis(cum_dist.real, argmax, axis=-1)
            + np.take_along_axis(cum_dist.imag, argmax, axis=-1)
        )
        / 2
    )
    rotational_angle_dict = {}
    threshold_dict = {}
    fidelity_dict = {}
    assignment_fidelity_dict = {}
    average_state0_dict = {}
    average_state1_dict = {}

    for qubit in qubits:
        rotational_angle_dict[qubit] = angle.flatten()
        threshold_dict[qubit] = threshold.flatten()
        fidelity_dict[qubit] = fidelity.flatten()
        assignment_fidelity_dict[qubit] = assignment_fidelity.flatten()
        average_state0_dict[qubit] = mean_gnd_state.flatten()
        average_state1_dict[qubit] = mean_exc_state.flatten()

    return RoFrequencyResults(
        rotational_angle_dict,
        threshold_dict,
        fidelity_dict,
        assignment_fidelity_dict,
        average_state0_dict,
        average_state1_dict,
    )


def _plot(data: RoFrequencyData, fit: RoFrequencyResults, qubit):
    """Plotting function for RoFrequency."""
    fig = go.Figure()

    # iterate over multiple data folders
    # subfolder = get_data_subfolders(folder)[0]
    # report_n = 0
    fitting_report = ""

    # Plot raw results with sliders
    for frequency in data.df["delta_frequency"].unique():
        state0_data = data.df[
            (data.df["delta_frequency"] == frequency) & (data.df["state"] == 0)
        ]
        state1_data = data.df[
            (data.df["delta_frequency"] == frequency) & (data.df["state"] == 1)
        ]
        fit_data = data_fit.df[data_fit.df["delta_frequency"] == frequency.magnitude]
        average_state0 = fit.average_state0[qubit]
        average_state1 = fit.average_state1[qubit]

        # print(fit_data)
        fig.add_trace(
            go.Scatter(
                x=state0_data["i"].pint.to("V").pint.magnitude,
                y=state0_data["q"].pint.to("V").pint.magnitude,
                name=f"q{qubit}/r{report_n}: state 0",
                mode="markers",
                showlegend=True,
                opacity=0.7,
                marker=dict(size=3, color=get_color_state0(report_n)),
                visible=False,
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=state1_data["i"].pint.to("V").pint.magnitude,
                y=state1_data["q"].pint.to("V").pint.magnitude,
                name=f"q{qubit}/r{report_n}: state 1",
                mode="markers",
                showlegend=True,
                opacity=0.7,
                marker=dict(size=3, color=get_color_state1(report_n)),
                visible=False,
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=average_state0[:, 0],
                y=average_state0[:, 1],
                name=f"q{qubit}/r{report_n}: mean state 0",
                showlegend=True,
                visible=False,
                mode="markers",
                marker=dict(size=10, color=get_color_state0(report_n)),
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=average_state1[:, 0],
                y=average_state1[:, 1],
                name=f"avg q{qubit}/r{report_n}: mean state 1",
                showlegend=True,
                visible=False,
                mode="markers",
                marker=dict(size=10, color=get_color_state1(report_n)),
            ),
        )

    # Show data for the first frequency
    for i in range(4):
        fig.data[i].visible = True

    # Add slider
    steps = []
    for i, freq in enumerate(data.df["frequency"].unique()):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},
            ],
            label=f"{freq:.6f}",
        )
        for j in range(4):
            step["args"][0]["visible"][i * 4 + j] = True
        steps.append(step)

    sliders = [
        dict(
            currentvalue={"prefix": "frequency: "},
            steps=steps,
        )
    ]

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="i (V)",
        yaxis_title="q (V)",
        sliders=sliders,
        title=f"q{qubit}",
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    # Plot the fidelity as a function of frequency
    fig_fidelity = go.Figure()

    fig_fidelity.add_trace(
        go.Scatter(x=data_fit.df["frequency"], y=data_fit.df["assignment_fidelity"])
    )
    fig_fidelity.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="delta frequency (Hz)",
        yaxis_title="assignment fidelity (ratio)",
        title=f"q{qubit}",
    )
    # Add fitting report for the best fidelity
    fit_data = data_fit.df[data_fit.df["fidelity"] == data_fit.df["fidelity"].max()]
    title_text = f"q{qubit}/r{report_n} | average state 0: ({complex(fit_data['average_state0'].to_numpy()[0]):.6f})<br>"
    title_text += f"q{qubit}/r{report_n} | average state 1: ({complex(fit_data['average_state1'].to_numpy()[0]):.6f})<br>"
    title_text += f"q{qubit}/r{report_n} | rotation angle: {float(fit_data['rotation_angle'].to_numpy()[0]):.3f} | threshold = {float(fit_data['threshold'].to_numpy()[0]):.6f}<br>"
    title_text += f"q{qubit}/r{report_n} | fidelity: {float(fit_data['fidelity'].to_numpy()[0]):.3f}<br>"
    title_text += f"q{qubit}/r{report_n} | assignment fidelity: {float(fit_data['assignment_fidelity'].to_numpy()[0]):.3f}<br>"
    title_text += f"q{qubit}/r{report_n} | optimal frequency: {float(fit_data['frequency'].to_numpy()[0]):.3f} Hz<br><br>"
    fitting_report = fitting_report + title_text
    return [fig, fig_fidelity], fitting_report

    return spectroscopy_plot(data, fit, qubit)


qubit_spectroscopy = Routine(_acquisition, _fit, _plot)
