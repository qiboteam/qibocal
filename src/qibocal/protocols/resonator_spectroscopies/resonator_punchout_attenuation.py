from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, Parameter, PulseSequence, Sweeper

from qibocal import update
from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.result import magnitude, phase

from qibolab._core.parameters import update_configs

from ..utils import HZ_TO_GHZ, fit_punchout, norm, table_dict, table_html

__all__ = ["resonator_punchout_attenuation", "ResonatorPunchoutAttenuationData"]


@dataclass
class ResonatorPunchoutAttenuationParameters(Parameters):
    """ResonatorPunchoutAttenuation runcard inputs."""

    freq_width: int
    """Width for frequency sweep relative to the readout frequency [Hz]."""
    freq_step: int
    """Frequency step for sweep [Hz]."""
    min_attenuation: float
    """Minimum LO attenuation [dB] (0 = no attenuation, higher = more attenuation)."""
    max_attenuation: float
    """Maximum LO attenuation [dB]."""
    step_attenuation: float
    """Step LO attenuation [dB]."""

@property
def attenuation_range(self) -> npt.NDArray[np.float64]:
    """LO attenuation range [dB]."""
    return np.arange(
        self.min_attenuation,
        self.max_attenuation + self.step_attenuation,
        self.step_attenuation,
    )


@dataclass
class ResonatorPunchoutAttenuationResults(Results):
    """ResonatorPunchoutAttenuation outputs."""

    readout_frequency: dict[QubitId, float]
    """Readout frequency [GHz] for each qubit."""
    bare_frequency: Optional[dict[QubitId, float]]
    """Bare resonator frequency [GHz] for each qubit."""
    readout_attenuation: dict[QubitId, float]
    """Readout LO attenuation [dB] for each qubit."""


ResPunchoutAttType = np.dtype(
    [
        ("freq", np.float64),
        ("attenuation", np.float64),
        ("signal", np.float64),
        ("phase", np.float64),
    ]
)
"""Custom dtype for resonator punchout attenuation."""


@dataclass
class ResonatorPunchoutAttenuationData(Data):
    """ResonatorPunchoutAttenuation data acquisition."""

    resonator_type: str
    """Resonator type."""
    attenuations: dict[QubitId, float] = field(default_factory=dict)
    """LO attenuations provided by the user."""
    data: dict[QubitId, npt.NDArray[ResPunchoutAttType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, freq: npt.NDArray[np.float64], attenuation: npt.NDArray[np.float64], signal: npt.NDArray[np.float64], phase: npt.NDArray[np.float64]):
        """Store output for single qubit."""
        size = len(freq) * len(attenuation)
        frequency, attenuation_grid = np.meshgrid(freq, attenuation)
        ar = np.empty(size, dtype=ResPunchoutAttType)
        ar["freq"] = frequency.ravel()
        ar["attenuation"] = attenuation_grid.ravel()
        ar["signal"] = signal.ravel()
        ar["phase"] = phase.ravel()
        self.data[qubit] = np.rec.array(ar)


def _acquisition(
    params: ResonatorPunchoutAttenuationParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> ResonatorPunchoutAttenuationData:
    """Data acquisition for Punchout using LO attenuation sweep.
    Arguments:
        params: ResonatorPunchoutAttenuationParameters
        platform: CalibrationPlatform
        targets: List of qubits to calibrate
    Returns:
        ResonatorPunchoutAttenuationData
    """

    # Define frequency range
    delta_frequency_range = np.arange(
        -params.freq_width / 2, params.freq_width / 2, params.freq_step
    )

    # Define attenuation range (0 = no attenuation, higher = more attenuation)
    

    # Get readout LO channels for each qubit
    ro_los = {}
    ro_pulses = {}
    original_attenuations = {}
    
    for qubit in targets:
        natives = platform.natives.single_qubit[qubit]
        ro_channel, ro_pulse = natives.MZ()[0]
        ro_pulses[qubit] = ro_pulse
        
        # Get the LO channel for this readout
        probe = platform.qubits[qubit].probe
        lo_channel = platform.channels[probe].lo
        ro_los[qubit] = lo_channel
        
        # Store original attenuation (power in qibocal is actually attenuation)
        original_attenuations[qubit] = platform.config(lo_channel).power

    data = ResonatorPunchoutAttenuationData(
        attenuations=original_attenuations,
        resonator_type=platform.resonator_type,
    )

    # Initialize data storage for all qubits
    all_signals = {qubit: [] for qubit in targets}
    all_phases = {qubit: [] for qubit in targets}

    try:
        # Loop over attenuation values
        for attenuation in attenuation_range:
            
            # Update LO attenuation for all target qubits
            updates = []
            for qubit in targets:
                lo_channel = ro_los[qubit]
                updates.append({lo_channel: {"power": attenuation}})

            # Create pulse sequence
            sequence = PulseSequence()
            freq_sweepers = {}
            
            for qubit in targets:
                natives = platform.natives.single_qubit[qubit]
                ro_channel, ro_pulse = natives.MZ()[0]
                sequence.append((ro_channel, ro_pulse))
                
                # Create frequency sweeper
                probe = platform.qubits[qubit].probe
                f0 = platform.config(probe).frequency
                freq_sweepers[qubit] = Sweeper(
                    parameter=Parameter.frequency,
                    values=f0 + delta_frequency_range,
                    channels=[probe],
                )
            
            # Execute with frequency sweep only
            results = platform.execute(
                [sequence],
                [[freq_sweepers[q] for q in targets]],
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
                averaging_mode=AveragingMode.CYCLIC,
                updates=updates,
            )
            
            # Collect results for this attenuation level
            for qubit in targets:
                ro_pulse = ro_pulses[qubit]
                result = list(results.items())[0][1]
                all_signals[qubit].append(magnitude(result))
                all_phases[qubit].append(phase(result))
    
    finally:
        pass
    
    # Register data for all qubits
    for qubit in targets:
        probe = platform.qubits[qubit].probe
        f0 = platform.config(probe).frequency
        frequencies = f0 + delta_frequency_range
        
        # Stack all attenuation sweeps
        signal_array = np.array(all_signals[qubit])
        phase_array = np.array(all_phases[qubit])
        
        data.register_qubit(
            qubit,
            signal=signal_array,
            phase=phase_array,
            freq=frequencies,
            attenuation=attenuation_range,
        )

    return data


def _fit(data: ResonatorPunchoutAttenuationData, fit_type="attenuation") -> ResonatorPunchoutAttenuationResults:
    """Fit frequency and attenuation at high and low power for a given resonator."""

    readout_freq, bare_freq, readout_param = fit_punchout(data, fit_type)
    
    return ResonatorPunchoutAttenuationResults(
        readout_frequency=readout_freq,
        bare_frequency=bare_freq,
        readout_attenuation=readout_param,
    )


def _plot(
    data: ResonatorPunchoutAttenuationData, 
    target: QubitId, 
    fit: ResonatorPunchoutAttenuationResults = None
):
    """Plotting function for ResonatorPunchoutAttenuation."""
    figures = []
    fitting_report = ""
    
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
        subplot_titles=(
            "Normalised Signal [a.u.]",
            "Phase [rad]",
        ),
    )
    
    qubit_data = data[target]
    frequencies = qubit_data.freq * HZ_TO_GHZ
    attenuations = qubit_data.attenuation
    n_attenuations = len(np.unique(qubit_data.attenuation))
    n_freq = len(np.unique(qubit_data.freq))
    
    # Normalize signal for each attenuation level
    for i in range(n_attenuations):
        qubit_data.signal[i * n_freq : (i + 1) * n_freq] = norm(
            qubit_data.signal[i * n_freq : (i + 1) * n_freq]
        )

    fig.add_trace(
        go.Heatmap(
            x=frequencies,
            y=attenuations,
            z=qubit_data.signal,
            colorbar_x=0.46,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            x=frequencies,
            y=attenuations,
            z=qubit_data.phase,
            colorbar_x=1.01,
        ),
        row=1,
        col=2,
    )
    
    # Flip y-axis so low attenuation (high power) is at top
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    
    if fit is not None:
        fig.add_trace(
            go.Scatter(
                x=[fit.readout_frequency[target] * HZ_TO_GHZ],
                y=[fit.readout_attenuation[target]],
                mode="markers",
                marker=dict(
                    size=8,
                    color="gray",
                    symbol="circle",
                ),
                name="Estimated readout point",
                showlegend=True,
            )
        )
        fitting_report = table_html(
            table_dict(
                target,
                [
                    "Low Power Resonator Frequency [Hz]",
                    "Low Power Readout LO Attenuation [dB]",
                    "High Power Resonator Frequency [Hz]",
                ],
                [
                    np.round(fit.readout_frequency[target]),
                    np.round(fit.readout_attenuation[target], 2),
                    np.round(fit.bare_frequency[target]) if fit.bare_frequency[target] is not None else "N/A",
                ],
            )
        )

    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h"),
    )

    fig.update_xaxes(title_text="Frequency [GHz]", row=1, col=1)
    fig.update_xaxes(title_text="Frequency [GHz]", row=1, col=2)
    fig.update_yaxes(title_text="LO Attenuation [dB]", row=1, col=1)

    figures.append(fig)

    return figures, fitting_report


def _update(
    results: ResonatorPunchoutAttenuationResults, 
    platform: CalibrationPlatform, 
    target: QubitId
):
    """Update platform with fitted parameters."""

    update.readout_frequency(results.readout_frequency[target], platform, target)
    if results.bare_frequency[target] is not None:
        update.bare_resonator_frequency(results.bare_frequency[target], platform, target)
        update.dressed_resonator_frequency(results.readout_frequency[target], platform, target)
    
    # Update LO attenuation (stored as power in qibocal)
    probe = platform.qubits[target].probe
    lo_channel = platform.channels[probe].lo

    platform.update(
        {lo_channel: {"power": results.readout_attenuation[target]}}
    )

def lo_frequency(freq: float, platform: Platform, qubit: QubitId, channel_type: literal["probe", "drive"] = "probe"):
    """Update LO frequency value in platform for specific qubit."""
    if channel_type not in ["probe", "drive"]:
        raise ValueError("channel_type must be either 'probe' or 'drive'")
    channel = getattr(platform.qubits[qubit], channel_type)
    platform.update({f"configs.{channel}.frequency": freq})


resonator_punchout_attenuation = Routine(_acquisition, _fit, _plot, _update)
"""ResonatorPunchoutAttenuation Routine object.

This routine performs a resonator punchout measurement by sweeping the LO attenuation
and frequency to determine the critical power for a qubit's resonator to be dispersively shifted.

At low power, the effective resonator frequency is shifted by χ due to the qubit state, while at high power, the resonator frequency
approaches its bare frequency.

.. math::
    \\omega_{dressed} = \\omega_r + \\chi

In general the frequency of the peak in the resonator can be approximated by 

.. math::
    \\omega_{\\text{peak}}(\\bar{n}) = \\omega_r + \\chi \\frac{1}{1 + \\bar{n}/n_{\\text{crit}}}

Where :math:`n_{\\text{crit}} = \\Delta^2 / 4g^2` is the critical photon number at which the resonator frequency starts to shift towards the bare frequency.
"""
