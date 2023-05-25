from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import DataUnits

from .resonator_spectroscopy import ResonatorSpectroscopyData
from .utils import lorentzian_fit, spectroscopy_plot


@dataclass
class QubitSpectroscopyParameters(Parameters):
    """QubitSpectroscopy runcard inputs."""

    freq_width: int
    """Width for frequency sweep relative  to the qubit frequency."""
    freq_step: int
    """Frequency step for sweep."""
    drive_duration: int
    """Drive pulse duration. Same for all qubits."""
    drive_amplitude: Optional[float] = None
    """Drive pulse amplitude (optional). Same for all qubits."""
    nshots: int = 1024
    """Number of shots."""
    relaxation_time: int = 50
    """Relaxation time (ns)."""


@dataclass
class QubitSpectroscopyResults(Results):
    """QubitSpectroscopy outputs."""

    frequency: Dict[List[Tuple], str] = field(metadata=dict(update="drive_frequency"))
    """Drive frequecy for each qubit."""
    amplitude: Dict[List[Tuple], str]
    """Input drive amplitude. Same for all qubits."""
    fitted_parameters: Dict[List[Tuple], List]
    """Raw fitting output."""
    attenuation: Optional[Dict[List[Tuple], str]] = field(
        default_factory=dict,
    )
    """Input attenuation for each qubit (optional)."""


class QubitSpectroscopyData(DataUnits):
    """QubitSpectroscopy acquisition outputs."""

    def __init__(
        self, resonator_type, power_level=None, amplitude=None, attenuation=None
    ):
        super().__init__(
            name="data",
            quantities={"frequency": "Hz"},
            options=[
                "probability",
            ],
        )
        self._resonator_type = resonator_type
        self._power_level = power_level
        self._amplitude = amplitude
        self._attenuation = attenuation

    @property
    def resonator_type(self):
        """Type of resonator"""
        return self._resonator_type

    @property
    def power_level(self):
        """Resonator spectroscopy power level"""
        return self._power_level

    @property
    def amplitude(self):
        """Readout pulse amplitude common for all qubits"""
        return self._amplitude

    @property
    def attenuation(self):
        """Attenuation value common for all qubits"""
        return self._attenuation


def _acquisition(
    params: QubitSpectroscopyParameters, platform: AbstractPlatform, qubits: Qubits
) -> QubitSpectroscopyData:
    """Data acquisition for qubit spectroscopy."""
    # create a sequence of pulses for the experiment:
    # long drive probing pulse - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=0, duration=params.drive_duration
        )
        if params.drive_amplitude is not None:
            qd_pulses[qubit].amplitude = params.drive_amplitude
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )
    sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[qd_pulses[qubit] for qubit in qubits],
    )

    # Create data structure for data acquisition.
    data = QubitSpectroscopyData(
        platform.resonator_type, amplitude=params.drive_amplitude
    )

    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        sweeper,
    )

    # retrieve the results for every qubit
    for qubit, ro_pulse in ro_pulses.items():
        # average msr, phase, i and q over the number of shots defined in the runcard
        result = results[ro_pulse.serial]
        r = result.serialize

        prob = np.abs(
            result.voltage_i
            + 1j * result.voltage_q
            - complex(platform.qubits[qubit].mean_gnd_states)
        ) / np.abs(
            complex(platform.qubits[qubit].mean_exc_states)
            - complex(platform.qubits[qubit].mean_gnd_states)
        )

        # store the results
        r.update(
            {
                "probability": prob,
                "frequency[Hz]": delta_frequency_range + qd_pulses[qubit].frequency,
                "qubit": len(delta_frequency_range) * [qubit],
            }
        )
        data.add_data_from_dict(r)
    return data


def _fit(data: QubitSpectroscopyData) -> QubitSpectroscopyResults:
    """Post-processing function for QubitSpectroscopy."""
    qubits = data.df["qubit"].unique()
    amplitudes = {}
    attenuations = {}
    frequency = {}
    fitted_parameters = {}
    for qubit in qubits:
        freq, fitted_params = lorentzian_fit(data, qubit)
        frequency[qubit] = freq
        amplitudes[qubit] = data.amplitude
        attenuations[qubit] = data.attenuation
        fitted_parameters[qubit] = fitted_params

    return QubitSpectroscopyResults(
        frequency=frequency,
        fitted_parameters=fitted_parameters,
        amplitude=amplitudes,
        attenuation=attenuations,
    )


def _plot(data: QubitSpectroscopyData, fit: QubitSpectroscopyResults, qubit):
    """Plotting function for QubitSpectroscopy."""
    figs, report = spectroscopy_plot(data, fit, qubit)

    data = data.df[data.df["qubit"] == qubit]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["frequency"].pint.to("GHz").pint.magnitude,
            y=data["probability"],
            opacity=1,
            name="Frequency",
            showlegend=True,
            legendgroup="Frequency",
        ),
    )
    fig.update_layout(
        title=f"Qubit {qubit} spectroscopy",
        xaxis_title="Frequency (GHz)",
        yaxis_title="Probability",
    )

    figs.append(fig)
    return figs, report


qubit_spectroscopy = Routine(_acquisition, _fit, _plot)
"""QubitSpectroscopy Routine object."""
