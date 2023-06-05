from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import DataUnits
from qibocal.plots.utils import get_color
from qibocal.protocols.characterization.resonator_spectroscopy import (
    ResonatorSpectroscopyData,
)
from qibocal.protocols.characterization.utils import (
    PowerLevel,
    lorentzian,
    lorentzian_fit,
)


@dataclass
class DispersiveShiftParameters(Parameters):
    """Dispersive shift inputs."""

    freq_width: int
    """Width [Hz] for frequency sweep relative to the readout frequency (Hz)."""
    freq_step: int
    """Frequency step for sweep (Hz)."""
    qubits: list = field(default_factory=list)
    """Local qubits (optional)."""
    update: Optional[bool] = None
    """Runcard update mechanism."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class StateResults(Results):
    """Resonator spectroscopy outputs."""

    frequency: Dict[Union[str, int], float]
    """Readout frequency for each qubit."""
    fitted_parameters: Dict[Union[str, int], Dict[str, float]]
    """Raw fitted parameters."""


@dataclass
class DispersiveShiftResults(Results):
    """Dispersive shift outputs."""

    results_0: StateResults
    """Resonator spectroscopy outputs in the ground state."""
    results_1: StateResults
    """Resonator spectroscopy outputs in the excited state"""
    best_freq: Dict[Union[str, int], float] = field(
        metadata=dict(update="readout_frequency")
    )
    """Readout frequency that maximizes the distance of ground and excited states in iq-plane"""
    best_iqs: Dict[Union[str, int], npt.NDArray[np.float64]]
    """iq-couples of ground and excited states with best frequency"""


class DispersiveShiftData(DataUnits):
    """Dipsersive shift acquisition outputs."""

    def __init__(self, resonator_type, power_level=PowerLevel.low):
        super().__init__(
            name="data",
            quantities={"frequency": "Hz"},
            options=["qubit", "state"],
        )
        self.resonator_type = resonator_type
        self.power_level = power_level


def _acquisition(
    params: DispersiveShiftParameters, platform: Platform, qubits: Qubits
) -> DispersiveShiftData:
    r"""
    Data acquisition for dispersive shift experiment.
    Perform spectroscopy on the readout resonator, with the qubit in ground and excited state, showing
    the resonator shift produced by the coupling between the resonator and the qubit.

    Args:
        params (DispersiveShiftParameters): experiment's parameters
        platform (Platform): Qibolab platform object
        qubits (dict): List of target qubits to perform the action

    """

    # create 2 sequences of pulses for the experiment:
    # sequence_0: I  - MZ
    # sequence_1: RX - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence_0 = PulseSequence()
    sequence_1 = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].duration
        )
        sequence_0.add(ro_pulses[qubit])
        sequence_1.add(qd_pulses[qubit])
        sequence_1.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )

    # create a DataUnits objects to store the results
    data = DispersiveShiftData(platform.resonator_type)

    sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[ro_pulses[qubit] for qubit in qubits],
    )

    results_0 = platform.sweep(
        sequence_0,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        sweeper,
    )

    results_1 = platform.sweep(
        sequence_1,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        sweeper,
    )

    # retrieve the results for every qubit
    for qubit in qubits:
        # average msr, phase, i and q over the number of shots defined in the runcard
        for i, results in enumerate([results_0, results_1]):
            result = results[ro_pulses[qubit].serial]
            # store the results
            r = result.serialize
            r.update(
                {
                    "frequency[Hz]": delta_frequency_range + ro_pulses[qubit].frequency,
                    "qubit": len(delta_frequency_range) * [qubit],
                    "state": len(delta_frequency_range) * [i],
                }
            )
            data.add_data_from_dict(r)

    return data


def _fit(data: DispersiveShiftData) -> DispersiveShiftResults:
    """Post-Processing for dispersive shift"""
    qubits = data.df["qubit"].unique()
    results = []
    iq_couples = [[], []]  # axis 0: states, axis 1: qubit

    for i in range(2):
        frequency = {}
        fitted_parameters = {}
        data_i = ResonatorSpectroscopyData(data.resonator_type)
        data_i.df = data.df[data.df["state"] == i].drop(columns=["state"]).reset_index()

        for qubit in qubits:
            freq, fitted_params = lorentzian_fit(data_i, qubit)
            frequency[qubit] = freq
            fitted_parameters[qubit] = fitted_params
            i_measures = data_i.df[data_i.df["qubit"] == qubit][
                "i"
            ].pint.magnitude.to_numpy()
            q_measures = data_i.df[data_i.df["qubit"] == qubit][
                "q"
            ].pint.magnitude.to_numpy()

            iq_couples[i].append(np.stack((i_measures, q_measures), axis=-1))
            results.append(StateResults(frequency, fitted_parameters))

    # for each qubit find the iq couple of 0-1 states that maximize the distance
    iq_couples = np.array(iq_couples)
    best_freqs = {}
    best_iqs = {}
    for qubit in qubits:
        frequencies = (
            data.df[data.df["qubit"] == qubit]["frequency"]
            .pint.to("GHz")
            .pint.magnitude.unique()
        )

        max_index = np.argmax(
            np.linalg.norm(iq_couples[0][qubit] - iq_couples[1][qubit], axis=-1)
        )
        best_freqs[qubit] = frequencies[max_index]
        best_iqs[qubit] = iq_couples[:, qubit, max_index]

    return DispersiveShiftResults(
        results_0=results[0],
        results_1=results[1],
        best_freq=best_freqs,
        best_iqs=best_iqs,
    )


def _plot(data: DispersiveShiftData, fit: DispersiveShiftResults, qubit):
    """Plotting function for dispersive shift."""
    figures = []

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
    # iterate over multiple data folders
    qubit_data = data.df[data.df["qubit"] == qubit]

    fitting_report = ""

    data_0 = ResonatorSpectroscopyData(data.resonator_type)
    data_0.df = qubit_data[qubit_data["state"] == 0].drop(columns=["state"])
    data_1 = ResonatorSpectroscopyData(data.resonator_type)
    data_1.df = (
        qubit_data[qubit_data["state"] == 1].drop(columns=["state"]).reset_index()
    )

    fit_data_0 = fit.results_0
    fit_data_1 = fit.results_1

    for i, label, q_data, data_fit in list(
        zip(
            (0, 1),
            ("State 0", "State 1"),
            (data_0, data_1),
            (fit_data_0, fit_data_1),
        )
    ):
        opacity = 1
        frequencies = q_data.df["frequency"].pint.to("GHz").pint.magnitude.unique()

        fig.add_trace(
            go.Scatter(
                x=q_data.df["frequency"].pint.to("GHz").pint.magnitude,
                y=q_data.df["MSR"].pint.to("uV").pint.magnitude,
                marker_color=get_color(3 * i),
                opacity=opacity,
                name=f"q{qubit}: {label}",
                showlegend=True,
                legendgroup=f"q{qubit}: {label}",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=q_data.df["frequency"].pint.to("GHz").pint.magnitude,
                y=q_data.df["phase"].pint.to("rad").pint.magnitude,
                marker_color=get_color(3 * i + 1),
                opacity=opacity,
                showlegend=False,
                legendgroup=f"q{qubit}: {label}",
            ),
            row=1,
            col=2,
        )

        freqrange = np.linspace(
            min(frequencies),
            max(frequencies),
            2 * len(data),
        )

        params = data_fit.fitted_parameters[qubit]
        fig.add_trace(
            go.Scatter(
                x=freqrange,
                y=lorentzian(freqrange, **params),
                name=f"q{qubit}: {label} Fit",
                line=go.scatter.Line(dash="dot"),
                marker_color=get_color(3 * i + 2),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=[fit.best_freq[qubit], fit.best_freq[qubit]],
            y=[
                np.min(qubit_data["MSR"].pint.to("uV").pint.magnitude),
                np.max(qubit_data["MSR"].pint.to("uV").pint.magnitude),
            ],
            mode="lines",
            line=go.scatter.Line(color="orange", width=3, dash="dash"),
            name="Best frequency",
        ),
        row=1,
        col=1,
    )

    fig.add_vline(
        x=fit.best_freq[qubit],
        line=dict(color="orange", width=3, dash="dash"),
        row=1,
        col=2,
    )

    fitting_report = fitting_report + (
        f"{qubit} | State zero freq : {fit_data_0.frequency[qubit]*1e9:,.0f} Hz.<br>"
    )
    fitting_report = fitting_report + (
        f"{qubit} | State one freq : {fit_data_1.frequency[qubit]*1e9:,.0f} Hz.<br>"
    )
    fitting_report = fitting_report + (
        f"{qubit} | Frequency shift : {(fit_data_1.frequency[qubit] - fit_data_0.frequency[qubit])*1e9:,.0f} Hz.<br>"
    )
    fitting_report = fitting_report + (
        f"{qubit} | Best frequency : {fit.best_freq[qubit]*1e9:,.0f} Hz.<br>"
    )
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Frequency (GHz)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Frequency (GHz)",
        yaxis2_title="Phase (rad)",
    )

    figures.append(fig)

    return figures, fitting_report


dispersive_shift = Routine(_acquisition, _fit, _plot)
"""Dispersive shift Routine object."""
