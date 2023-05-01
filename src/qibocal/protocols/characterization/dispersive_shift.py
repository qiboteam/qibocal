from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from scipy.optimize import curve_fit

from ...auto.operation import Parameters, Qubits, Results, Routine
from ...data import DataUnits
from ...plots.utils import get_color
from .resonator_spectroscopy import ResonatorSpectroscopyData
from .utils import PowerLevel, lorentzian, lorentzian_fit, spectroscopy_plot


@dataclass
class DispersiveShiftParameters(Parameters):
    freq_width: int
    freq_step: int


@dataclass
class StateResults(Results):
    frequency: Dict[List[Tuple], str]
    fitted_parameters: Dict[List[Tuple], List]
    # bare_frequency: Optional[Dict[List[Tuple], str]]
    amplitude: Optional[Dict[List[Tuple], str]]
    # attenuation: Optional[Dict[List[Tuple], str]]


@dataclass
class DispersiveShiftResults(Results):
    results_0: StateResults
    results_1: StateResults


class DispersiveShiftData(DataUnits):
    def __init__(
        self,
        resonator_type,
        power_level=PowerLevel.low,
        amplitude=None,
        attenuation=None,
    ):
        super().__init__(
            name="data",
            quantities={"frequency": "Hz"},
            options=["qubit", "iteration", "state"],
        )
        self.resonator_type = resonator_type
        self._power_level = power_level
        self._amplitude = amplitude
        self._attenuation = attenuation


def _acquisition(
    params: DispersiveShiftParameters, platform: AbstractPlatform, qubits: Qubits
) -> DispersiveShiftData:
    r"""
    Perform spectroscopy on the readout resonator, with the qubit in ground and excited state, showing
    the resonator shift produced by the coupling between the resonator and the qubit.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (dict): List of target qubits to perform the action
        freq_width (int): Width frequency in HZ to perform the spectroscopy sweep
        freq_step (int): Step frequency in HZ for the spectroscopy sweep
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Returns:
        A DataUnits object with the raw data obtained for the normal and shifted sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **frequency[Hz]**: Resonator frequency value in Hz
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages
    """
    # TODO: add sweepers
    # reload instrument settings from runcard
    # platform.reload_settings()

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

    # TODO: implement sweeper

    for delta_freq in delta_frequency_range:
        # reconfigure the instruments based on the new resonator frequency
        # in this case setting the local oscillators
        # the pulse sequence does not need to be modified or recreated between executions
        for qubit in qubits:
            ro_pulses[qubit].frequency = delta_freq + qubits[qubit].readout_frequency

        results_0 = platform.execute_pulse_sequence(sequence_0)
        results_1 = platform.execute_pulse_sequence(sequence_1)

        # retrieve the results for every qubit
        for i, results in enumerate([results_0, results_1]):
            for ro_pulse in ro_pulses.values():
                # average msr, phase, i and q over the number of shots defined in the runcard
                r = results[ro_pulse.serial].average.raw
                # store the results
                r.update(
                    {
                        "frequency[Hz]": ro_pulse.frequency,
                        "qubit": ro_pulse.qubit,
                        "state": i,
                    }
                )
                data.add_data_from_dict(r)

    return data


def _fit(data: DispersiveShiftData) -> DispersiveShiftResults:
    data_0 = ResonatorSpectroscopyData(data.resonator_type)
    data_0.df = data.df[data.df["state"] == 0].drop(columns=["state"]).reset_index()

    data_1 = ResonatorSpectroscopyData(data.resonator_type)
    data_1.df = data.df[data.df["state"] == 1].drop(columns=["state"]).reset_index()

    results_0 = StateResults(**lorentzian_fit(data_0))
    results_1 = StateResults(**lorentzian_fit(data_1))
    return DispersiveShiftResults(results_0, results_1)


def _plot(data: DispersiveShiftData, fit: DispersiveShiftResults, qubit):
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

    qubit_data = data.df[data.df["qubit"] == qubit].reset_index()
    report_n = 0
    fitting_report = ""

    data_0 = ResonatorSpectroscopyData(data.resonator_type)
    data_0.df = (
        qubit_data[qubit_data["state"] == 0].drop(columns=["state"]).reset_index()
    )

    data_1 = ResonatorSpectroscopyData(data.resonator_type)
    data_1.df = (
        qubit_data[qubit_data["state"] == 1].drop(columns=["state"]).reset_index()
    )

    fit_data_0 = fit.results_0
    fit_data_1 = fit.results_1

    resonator_freqs = {}
    for i, label, data, data_fit in list(
        zip(
            (0, 1),
            ("Spectroscopy", "Shifted spectroscopy"),
            (data_0, data_1),
            (fit_data_0, fit_data_1),
        )
    ):
        frequencies = data.df["frequency"].pint.to("GHz").pint.magnitude.unique()

        opacity = 1

        fig.add_trace(
            go.Scatter(
                x=data.df["frequency"].pint.to("GHz").pint.magnitude,
                y=data.df["MSR"].pint.to("uV").pint.magnitude,
                marker_color=get_color(2 * report_n + i),
                opacity=opacity,
                name=f"q{qubit}/r{report_n}: {label}",
                showlegend=True,
                legendgroup=f"q{qubit}/r{report_n}: {label}",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.df["frequency"].pint.to("GHz").pint.magnitude,
                y=data.df["phase"].pint.to("rad").pint.magnitude,
                marker_color=get_color(2 * report_n + i),
                opacity=opacity,
                showlegend=False,
                legendgroup=f"q{qubit}/r{report_n}: {label}",
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
                name=f"q{qubit}/r{report_n}: {label} Fit",
                line=go.scatter.Line(dash="dot"),
                marker_color=get_color(3 * report_n + i),
            ),
            row=1,
            col=1,
        )

    fitting_report = fitting_report + (
        f"q{qubit}/r{report_n} | state zero freq : {fit_data_0.frequency[qubit]:,.0f} Hz.<br>"
    )
    fitting_report = fitting_report + (
        f"q{qubit}/r{report_n} | state one freq : {fit_data_1.frequency[qubit]:,.0f} Hz.<br>"
    )
    fitting_report = fitting_report + (
        f"q{qubit}/r{report_n} | frequency shift : {fit_data_1.frequency[qubit] - fit_data_0.frequency[qubit]:,.0f} Hz.<br>"
    )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Frequency (Hz)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Frequency (Hz)",
        yaxis2_title="Phase (rad)",
    )

    figures.append(fig)

    return figures, fitting_report


dispersive_shift = Routine(_acquisition, _fit, _plot)
