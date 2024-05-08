from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import Qubit, QubitId

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.protocols.characterization.utils import table_dict, table_html


@dataclass
class QubitSpectroscopyParameters(Parameters):
    """QubitSpectroscopy runcard inputs."""

    freq_width: int
    """Width [Hz] for frequency sweep relative  to the qubit frequency."""
    freq_step: int
    """Frequency [Hz] step for sweep."""
    drive_duration: int
    """Drive pulse duration [ns]. Same for all qubits."""
    drive_amplitude: Optional[float] = None
    """Drive pulse duration [ns]. Same for all qubits."""
    drive_port_attenuation: Optional[int] = None
    """Drive pulse amplitude (optional). Same for all qubits."""
    intermediate_freq: Optional[float] = None


@dataclass
class QubitSpectroscopyResults(Results):
    """QubitSpectroscopy outputs."""

    frequency: dict[QubitId, dict[str, float]]
    """Drive frequecy [GHz] for each qubit."""
    amplitude: dict[QubitId, float]
    """Input drive amplitude. Same for all qubits."""
    fitted_parameters: dict[QubitId, list[float]]
    """Raw fitting output."""


QubitSpectroscopyType = np.dtype(
    [("freq", np.float64), ("signal", np.float64), ("phase", np.float64)]
)


@dataclass
class QubitSpectroscopyData(Data):
    """QubitSpectroscopy acquisition outputs."""

    resonator_type: str
    """Resonator type."""
    amplitudes: dict[QubitId, float]
    """Drive pulse amplitudes."""
    port_attenuations: dict[QubitId, float]
    """Port attenuations."""
    intermediate_freqs: float
    drive_freqs: float

    data: dict[QubitId, npt.NDArray[QubitSpectroscopyType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""


def _acquisition(
    params: QubitSpectroscopyParameters, platform: Platform, qubits: Qubits
) -> QubitSpectroscopyData:
    """Data acquisition for qubit spectroscopy."""
    # create a sequence of pulses for the experiment:
    # long drive probing pulse - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    ro_pulses = {}
    qd_pulses = {}
    amplitudes = {}
    port_attenuations = {}
    intermediate_freqs = {}
    drive_freqs = {}

    # Create data structure for data acquisition.
    data = QubitSpectroscopyData(
        resonator_type=platform.resonator_type,
        amplitudes=amplitudes,
        port_attenuations=port_attenuations,
        intermediate_freqs=intermediate_freqs,
        drive_freqs=drive_freqs,
    )
    for qubit_id in qubits:
        qubit: Qubit = qubits[qubit_id]
        sequence = PulseSequence()
        qd_pulses[qubit_id] = platform.create_qubit_drive_pulse(
            qubit_id, start=0, duration=params.drive_duration
        )
        if params.drive_amplitude is not None:
            qd_pulses[qubit_id].amplitude = params.drive_amplitude
        if params.drive_port_attenuation is not None:
            qubit.drive.attenuation = max(
                min((params.drive_port_attenuation // 2) * 2, 60), 0
            )

        amplitudes[qubit_id] = qd_pulses[qubit_id].amplitude
        port_attenuations[qubit_id] = qubit.drive.attenuation

        ro_pulses[qubit_id] = platform.create_qubit_readout_pulse(
            qubit_id, start=qd_pulses[qubit_id].finish
        )
        sequence.add(qd_pulses[qubit_id])
        sequence.add(ro_pulses[qubit_id])

        # drive_lo_frequency = qubit.drive.lo_frequency
        # drive_intermediate_frequency = qubit.drive_frequency - drive_lo_frequency
        drive_freqs[qubit_id] = qubit.drive_frequency

        if params.intermediate_freq is not None:
            intermediate_freqs[qubit_id] = params.intermediate_freq
        else:
            # select an intermediate frequency between 20 and 300 MHz, multiple of 20MHz
            intermediate_freqs[qubit_id] = max(
                min(((params.freq_width // 3) // 20e6) * 20e6, 300e6), 20e6
            )

        # define the parameter to sweep and its range:
        frequency_range_low_res = np.arange(
            -params.freq_width // 2, params.freq_width // 2, params.freq_step
        )
        frequency_range_high_res = np.arange(
            -params.freq_width // 40, params.freq_width // 40, params.freq_step // 4
        )
        frequency_range = np.unique(
            np.concatenate(
                (
                    frequency_range_low_res + drive_freqs[qubit_id],
                    frequency_range_high_res
                    + drive_freqs[qubit_id]
                    - intermediate_freqs[qubit_id],
                    frequency_range_high_res + drive_freqs[qubit_id],
                    frequency_range_high_res
                    + drive_freqs[qubit_id]
                    + intermediate_freqs[qubit_id],
                )
            )
        )

        for frequency in frequency_range:
            qubit.drive.lo_frequency = frequency
            qd_pulses[qubit_id].frequency = frequency + intermediate_freqs[qubit_id]

            # execute the pulse sequence
            results = platform.execute_pulse_sequence(
                sequence,
                ExecutionParameters(
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    acquisition_type=AcquisitionType.INTEGRATION,
                    averaging_mode=AveragingMode.CYCLIC,
                ),
            )

            result = results[ro_pulses[qubit_id].serial]
            data.register_qubit(
                QubitSpectroscopyType,
                (qubit_id),
                dict(
                    freq=np.array([frequency]),
                    signal=np.array([result.magnitude]),
                    phase=np.array([result.phase]),
                ),
            )
    data.amplitudes = amplitudes
    data.port_attenuations = port_attenuations
    data.intermediate_freqs = intermediate_freqs
    data.drive_freqs = drive_freqs
    return data


def _fit(data: QubitSpectroscopyData) -> QubitSpectroscopyResults:
    """Post-processing function for QubitSpectroscopy."""
    return QubitSpectroscopyResults(None, None, None)


def _plot(data: QubitSpectroscopyData, qubit, fit: QubitSpectroscopyResults):
    """Plotting function for QubitSpectroscopy."""
    HZ_TO_GHZ = 1e-9
    figures = []
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
    )
    qubit_data = data[qubit]
    fitting_report = ""

    frequencies = qubit_data.freq * HZ_TO_GHZ
    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=qubit_data.signal,
            opacity=1,
            name="Frequency",
            showlegend=True,
            legendgroup="Frequency",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=qubit_data.phase,
            opacity=1,
            name="Phase",
            showlegend=True,
            legendgroup="Phase",
        ),
        row=1,
        col=2,
    )

    # add vertical lines at the point where the qubit is expected and its image and LO
    lo_pif = data.drive_freqs[qubit] - data.intermediate_freqs[qubit]
    lo_mif = data.drive_freqs[qubit] + data.intermediate_freqs[qubit]
    lo = data.drive_freqs[qubit]
    for f, c, n in [
        (lo_mif, "coral", "image"),
        (lo, "coral", "leakage"),
        (lo_pif, "skyblue", "signal"),
    ]:
        fig.add_vline(
            type="line",
            x=f * HZ_TO_GHZ,
            line=dict(color=c, width=1, dash="dash"),
            annotation_text=n,
            annotation_font_color="grey",
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Frequency [GHz]",
        yaxis_title="Signal [a.u.]",
        xaxis2_title="Frequency [GHz]",
        yaxis2_title="Phase [rad]",
    )
    figures.append(fig)
    fitting_report = table_html(
        table_dict(
            qubit,
            [
                "Intermediate Frequency",
                "Amplitude",
                "Port Attenuation",
            ],
            [
                f"{data.intermediate_freqs[qubit]:,.0f}",
                f"{data.amplitudes[qubit]:,.3f}",
                f"{data.port_attenuations[qubit]}",
            ],
        )
    )
    return figures, fitting_report


def _update(results: QubitSpectroscopyResults, platform: Platform, qubit: QubitId):
    # update.drive_frequency(results.frequency[qubit], platform, qubit)
    pass


qubit_spectroscopy_with_lo = Routine(_acquisition, _fit, _plot, _update)
"""QubitSpectroscopy Routine object."""
