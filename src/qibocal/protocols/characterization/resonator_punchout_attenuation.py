from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.config import log
from qibocal.data import DataUnits

from .utils import find_min_msr, get_max_freq, get_points_with_max_freq, norm


@dataclass
class ResonatorPunchoutAttenuationParameters(Parameters):
    """ResonatorPunchoutAttenuation runcard inputs."""

    freq_width: int
    """Width for frequency sweep relative  to the readout frequency (Hz)."""
    freq_step: int
    """Frequency step for sweep (Hz)."""
    min_att: int
    """Attenuation minimum value (dB)."""
    max_att: int
    """Attenuation maximum value (dB)."""
    step_att: int
    """Attenuation step (dB)."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""
    qubits: Optional[list] = field(default_factory=list)
    """Local qubits (optional)."""


@dataclass
class ResonatorPunchoutAttenuationResults(Results):
    """ResonatorPunchoutAttenation outputs."""

    readout_frequency: Dict[Union[str, int], float] = field(
        metadata=dict(update="readout_frequency")
    )
    """Readout frequency [GHz] for each qubit."""
    readout_attenuation: Dict[Union[str, int], int] = field(
        metadata=dict(update="readout_attenuation")
    )
    """Readout attenuation [dB] for each qubit."""
    bare_frequency: Optional[Dict[Union[str, int], float]] = field(
        metadata=dict(update="bare_resonator_frequency")
    )
    """Bare resonator frequency [GHz] for each qubit."""
    lp_max_att: Dict[Union[str, int], int]
    """Maximum attenuation at low power for each qubit."""
    lp_min_att: Dict[Union[str, int], int]
    """Minimum attenuation at low power for each qubit."""
    hp_max_att: Dict[Union[str, int], int]
    """Maximum attenuation at high power for each qubit."""
    hp_min_att: Dict[Union[str, int], int]
    """Minimum attenuation at high power for each qubit."""


class ResonatorPunchoutAttenuationData(DataUnits):
    """ResonatorPunchoutAttenuation data acquisition."""

    def __init__(self, resonator_type):
        super().__init__(
            "data",
            {"frequency": "Hz", "attenuation": "dB"},
            options=[
                "qubit",
            ],
        )
        self._resonator_type = resonator_type

    @property
    def resonator_type(self):
        """Type of resonator (2D or 3D)."""
        return self._resonator_type


def _acquisition(
    params: ResonatorPunchoutAttenuationParameters,
    platform: AbstractPlatform,
    qubits: Qubits,
) -> ResonatorPunchoutAttenuationData:
    """Data acquisition for Punchout over attenuation."""
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

    # attenuation
    attenuation_range = np.arange(params.min_att, params.max_att, params.step_att)
    att_sweeper = Sweeper(
        Parameter.attenuation, attenuation_range, qubits=list(qubits.values())
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include resonator frequency and attenuation
    data = ResonatorPunchoutAttenuationData(platform.resonator_type)

    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        att_sweeper,
        freq_sweeper,
    )

    # retrieve the results for every qubit
    for qubit in qubits:
        # average msr, phase, i and q over the number of shots defined in the runcard
        result = results[ro_pulses[qubit].serial]
        att = np.repeat(attenuation_range, len(delta_frequency_range))
        # store the results
        freqs = np.array(
            len(attenuation_range)
            * list(delta_frequency_range + ro_pulses[qubit].frequency)
        ).flatten()
        r = {k: v.ravel() for k, v in result.serialize.items()}
        r.update(
            {
                "frequency[Hz]": freqs,
                "attenuation[dB]": att,
                "qubit": len(freqs) * [qubit],
            }
        )
        data.add_data_from_dict(r)

        # # Temporary fixe to force to reset the attenuation to the original value in qblox
        # # sweeper method returning to orig value not working for attenuation
        # # After fitting punchout the reload_settings will be called automatically
        # platform.reload_settings()
        # save data
    return data


def _fit(
    data: ResonatorPunchoutAttenuationData, fit_type="attenuation"
) -> ResonatorPunchoutAttenuationResults:
    """Fit frequency and attenuation at high and low power for a given resonator."""

    qubits = data.df["qubit"].unique()

    freq_lp_dict = {}
    lp_max_att_dict = {}
    lp_min_att_dict = {}
    freq_hp_dict = {}
    hp_max_att_dict = {}
    hp_min_att_dict = {}
    ro_att_dict = {}

    for qubit in qubits:
        averaged_data = (
            data.df[data.df["qubit"] == qubit]
            .drop(columns=["i", "q", "qubit"])
            .groupby(["frequency", fit_type], as_index=False)
            .mean()
        )
        try:
            normalised_data = averaged_data.groupby([fit_type], as_index=False)[
                ["MSR"]
            ].transform(norm)

            averaged_data_updated = averaged_data.copy()
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
            lp_max_att = point_lp_max[1]
            lp_min_att = point_lp_min[1]
            freq_hp = point_hp_max[0]
            hp_max_att = point_hp_max[1]
            hp_min_att = point_hp_min[1]
            ro_att = round((lp_max_att + lp_min_att) / 2)
            ro_att = ro_att + 1 if ro_att % 2 == 1 else ro_att

        except:
            log.warning("resonator_punchout_fit: the fitting was not succesful")
            freq_lp = 0.0
            freq_hp = 0.0
            ro_att = 0.0
            lp_max_att = 0.0
            lp_min_att = 0.0
            hp_max_att = 0.0
            hp_min_att = 0.0

        freq_lp_dict[qubit] = freq_lp / 1e9
        freq_hp_dict[qubit] = freq_hp / 1e9
        ro_att_dict[qubit] = ro_att
        lp_max_att_dict[qubit] = lp_max_att
        lp_min_att_dict[qubit] = lp_min_att
        hp_max_att_dict[qubit] = hp_max_att
        hp_min_att_dict[qubit] = hp_min_att

    return ResonatorPunchoutAttenuationResults(
        freq_lp_dict,
        ro_att_dict,
        freq_hp_dict,
        lp_max_att_dict,
        lp_min_att_dict,
        hp_max_att_dict,
        hp_min_att_dict,
    )


def _plot(
    data: ResonatorPunchoutAttenuationData,
    fit: ResonatorPunchoutAttenuationResults,
    qubit,
):
    """Plotting for ResonatorPunchoutAttenuation."""

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

    # TODO: remove this function
    def norm(x):
        x_mags = x.pint.to("V").pint.magnitude
        return (x_mags - np.min(x_mags)) / (np.max(x_mags) - np.min(x_mags))

    qubit_data = data.df[data.df["qubit"] == qubit].drop(columns=["qubit"])

    normalised_data = qubit_data.groupby(["attenuation"], as_index=False)[
        ["MSR"]
    ].transform(norm)

    fig.add_trace(
        go.Heatmap(
            x=qubit_data["frequency"].pint.to("Hz").pint.magnitude,
            y=qubit_data["attenuation"].pint.to("dB").pint.magnitude,
            z=normalised_data["MSR"],
            colorbar_x=0.46,
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text=f"{qubit}: Frequency (Hz)", row=1, col=1)
    fig.update_yaxes(title_text="Attenuation", row=1, col=1)
    fig.add_trace(
        go.Heatmap(
            x=qubit_data["frequency"].pint.to("Hz").pint.magnitude,
            y=qubit_data["attenuation"].pint.to("dB").pint.magnitude,
            z=qubit_data["phase"].pint.to("rad").pint.magnitude,
            colorbar_x=1.01,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text=f"{qubit}/: Frequency (Hz)", row=1, col=2)
    fig.update_yaxes(title_text="Attenuation", row=1, col=2)

    if len(data) > 0:
        fig.add_trace(
            go.Scatter(
                x=[
                    fit.readout_frequency[qubit] * 1e9,
                    fit.readout_frequency[qubit] * 1e9,
                    fit.bare_frequency[qubit] * 1e9,
                    fit.bare_frequency[qubit] * 1e9,
                ],
                y=[
                    fit.lp_max_att[qubit],
                    fit.lp_min_att[qubit],
                    fit.hp_max_att[qubit],
                    fit.hp_min_att[qubit],
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
        title_text += f"{qubit} | Resonator Frequency at Low Power:  {fit.readout_frequency[qubit] * 1e9} Hz.<br>"
        title_text += f"{qubit} | Low Power Attenuation Range: {fit.lp_max_att[qubit]} - {fit.lp_min_att[qubit]} db.<br>"
        title_text += f"{qubit} | Resonator Frequency at High Power: {fit.bare_frequency[qubit] * 1e9} Hz.<br>"
        title_text += f"{qubit} | High Power Attenuation Range: {fit.hp_max_att[qubit]} - {fit.hp_min_att[qubit]} db.<br>"

        fitting_report = fitting_report + title_text
    else:
        fitting_report = "No fitting data"

    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
    )

    figures.append(fig)

    return figures, fitting_report


resonator_punchout_attenuation = Routine(_acquisition, _fit, _plot)
"""ResonatorPunchoutAttenuation Routine object."""
