from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from ...auto.operation import Parameters, Qubits, Results, Routine
from ...config import log
from ...data import DataUnits
from .utils import find_min_msr, get_max_freq, get_points_with_max_freq, norm


@dataclass
class ResonatorPunchoutParameters(Parameters):
    freq_width: int
    freq_step: int
    min_amp_factor: float
    max_amp_factor: float
    step_amp_factor: float
    nshots: int
    relaxation_time: int


@dataclass
class ResonatorPunchoutResults(Results):
    readout_frequency: Dict[List[Tuple], str] = field(
        metadata=dict(update="readout_frequency")
    )
    readout_amplitude: Dict[List[Tuple], str] = field(
        metadata=dict(update="readout_amplitude")
    )
    bare_frequency: Optional[Dict[List[Tuple], str]] = field(
        metadata=dict(update="bare_resonator_frequency")
    )


class ResonatorPunchoutData(DataUnits):
    def __init__(self, resonator_type=None):
        super().__init__(
            "data",
            {"frequency": "Hz", "amplitude": "dimensionless"},
            options=["qubit"],
        )
        self._resonator_type = resonator_type

    @property
    def resonator_type(self):
        return self._resonator_type


def _acquisition(
    platform: AbstractPlatform, qubits: Qubits, params: ResonatorPunchoutParameters
) -> ResonatorPunchoutData:
    # create a sequence of pulses for the experiment:
    # MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()

    ro_pulses = {}
    for qubit in qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulses[qubit])

    # define the parameters to sweep and their range:
    # resonator frequency
    delta_frequency_range = np.arange(
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )
    freq_sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        [ro_pulses[qubit] for qubit in qubits],
    )

    # amplitude
    amplitude_range = np.arange(
        params.min_amp_factor, params.max_amp_factor, params.step_amp_factor
    )
    amp_sweeper = Sweeper(
        Parameter.amplitude, amplitude_range, [ro_pulses[qubit] for qubit in qubits]
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include resonator frequency and attenuation
    data = ResonatorPunchoutData(platform.resonator_type)

    # repeat the experiment as many times as defined by software_averages
    amps = np.repeat(amplitude_range, len(delta_frequency_range))

    results = platform.sweep(
        sequence,
        amp_sweeper,
        freq_sweeper,
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
    )

    # retrieve the results for every qubit
    for qubit, ro_pulse in ro_pulses.items():
        # average msr, phase, i and q over the number of shots defined in the runcard
        result = results[ro_pulse.serial]
        # store the results
        freqs = np.array(
            len(amplitude_range) * list(delta_frequency_range + ro_pulse.frequency)
        ).flatten()
        r = {k: v.ravel() for k, v in result.raw.items()}
        r.update(
            {
                "frequency[Hz]": freqs,
                "amplitude[dimensionless]": amps,
                "qubit": len(freqs) * [qubit],
            }
        )
        data.add_data_from_dict(r)

    return data


def _fit(data: ResonatorPunchoutData, fit_type="amplitude") -> ResonatorPunchoutResults:
    # def punchout_fit(data: DataUnits, fit_type: str) -> Results:
    # def punchout_fit(data, qubits, resonator_type, labels, fit_type):
    """Fit frequency and attenuation at high and low power for a given resonator
        Args:
        data (DataUnits): data file with information on the feature response at each current point.
        qubits (list): qubits coupled to the resonator that we are probing.
        resonator_type (str): the type of readout resonator ['3D', '2D'].
        labels (list of str): list containing the lables of the quantities computed by this fitting method.
        fit_type (str): the type of punchout executed ['attenuation', 'amplitude'].

    Returns:
        data_fit (Data): Data file with labels and fit parameters (frequency at low and high power, attenuation range for low and highg power)
    """

    qubits = data.df["qubit"].unique()

    bare_freqs = {}
    dressed_freqs = {}
    ro_amplitudes = {}

    for qubit in qubits:
        qubit_data = data.df[data.df["qubit"] == qubit].drop(
            columns=["i", "q", "qubit"]
        )

        try:
            normalised_data = qubit_data.groupby([fit_type], as_index=False)[
                ["MSR"]
            ].transform(norm)

            averaged_data_updated = qubit_data.copy()
            averaged_data_updated.update(normalised_data["MSR"])

            min_points = find_min_msr(
                averaged_data_updated, data.resonator_type, fit_type
            )
            vfunc = np.vectorize(lambda x: x.magnitude)
            min_points = vfunc(min_points)

            max_x = np.amax(min_points[:, 0])
            min_x = np.amin(min_points[:, 0])
            middle_x = (max_x + min_x) / 2

            hp_points = min_points[min_points[:, 0] < middle_x]
            lp_points = min_points[min_points[:, 0] >= middle_x]

            freq_hp = get_max_freq(hp_points)
            freq_lp = get_max_freq(lp_points)

            point_hp_max, point_hp_min = get_points_with_max_freq(min_points, freq_hp)
            point_lp_max, point_lp_min = get_points_with_max_freq(min_points, freq_lp)

            freq_lp = point_lp_max[0]
            ro_amp = point_lp_max[1]
            freq_hp = point_hp_max[0]

        except:
            log.warning("resonator_punchout_fit: the fitting was not succesful")
            freq_lp = 0.0
            freq_hp = 0.0
            ro_amp = 0.0

        dressed_freqs[qubit] = freq_lp / 1e9
        bare_freqs[qubit] = freq_hp / 1e9
        ro_amplitudes[qubit] = ro_amp

    return ResonatorPunchoutResults(
        dressed_freqs,
        ro_amplitudes,
        bare_freqs,
    )


def _plot(data: ResonatorPunchoutData, fit: ResonatorPunchoutResults, qubit):
    figures = []
    fitting_report = ""

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
        subplot_titles=(
            "Normalised MSR",
            "phase (rad)",
        ),
    )

    qubit_data = data.df[data.df["qubit"] == qubit].drop(columns=["qubit"])

    normalised_data = qubit_data.groupby(["amplitude"], as_index=False)[
        ["MSR"]
    ].transform(norm)

    fig.add_trace(
        go.Heatmap(
            x=qubit_data["frequency"].pint.to("Hz").pint.magnitude,
            y=qubit_data["amplitude"].pint.to("dimensionless").pint.magnitude,
            z=normalised_data["MSR"].pint.to("dimensionless").pint.magnitude,
            colorbar_x=0.46,
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text=f"{qubit}: Frequency (Hz)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.add_trace(
        go.Heatmap(
            x=qubit_data["frequency"].pint.to("Hz").pint.magnitude,
            y=qubit_data["amplitude"].pint.to("dimensionless").pint.magnitude,
            z=qubit_data["phase"].pint.to("rad").pint.magnitude,
            colorbar_x=1.01,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=[
                fit.readout_frequency[qubit] * 1e9,
            ],
            y=[
                fit.readout_amplitude[qubit],
            ],
            mode="markers",
            marker=dict(
                size=8,
                color="gray",
                symbol="circle",
            ),
        )
    )
    title_text = ""
    title_text += f"{qubit} | Resonator Frequency at Low Power:  {fit.readout_frequency[qubit]*1e9:,.0f} Hz<br>"
    title_text += f"{qubit} | Resonator Frequency at High Power: {fit.bare_frequency[qubit]*1e9:,.0f} Hz<br>"
    title_text += f"{qubit} | Readout Amplitude at Low Power: {fit.readout_amplitude[qubit]:,.3f} <br>"

    fitting_report = fitting_report + title_text

    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
    )

    figures.append(fig)

    return figures, fitting_report


resonator_punchout = Routine(_acquisition, _fit, _plot)
