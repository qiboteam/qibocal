"""Interleaved decoherence protocol for T1 and Ramsey measurements.
"""

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional, Dict, List, Tuple, Union
import csv

import numpy as np
import rich
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import Delay, PulseSequence, Sweeper, Parameter, Platform
from qibolab._core.native import SingleQubitNatives
from qibolab._core.execution_parameters import AveragingMode

from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine

import argparse

GHz = 1e9

__all__ = [
    "InterleavedDecoherenceParameters",
    "InterleavedDecoherenceResults",
    "InterleavedDecoherenceData",
    "InterleavedDecoherenceType",
    "interleaved_decoherence",
]


# Custom dtype for interleaved decoherence data
InterleavedDecoherenceType = np.dtype(
    [
        ("delay", np.float64),
        ("prob", np.float64),
        ("error", np.float64),
        ("experiment", "U10"),  # "t1" or "ramsey"
    ]
)


@dataclass
class InterleavedDecoherenceParameters(Parameters):
    """Interleaved decoherence protocol parameters.
    
    Attributes:
        t1_delay_start: Initial delay for T1 measurement [ns].
        t1_delay_end: Final delay for T1 measurement [ns].
        t1_delay_step: Step for T1 delay sweep [ns].
        ramsey_delay_start: Initial delay for Ramsey measurement [ns].
        ramsey_delay_end: Final delay for Ramsey measurement [ns].
        ramsey_delay_step: Step for Ramsey delay sweep [ns].
        ramsey_detuning: Frequency detuning for Ramsey [Hz]. If 0, standard Ramsey.
        nshots: Number of measurement shots.
        relaxation_time: Wait time between measurements [ns].
    """
    
    t1_delay_start: int = 80
    """Initial delay for T1 measurement [ns]."""
    t1_delay_end: int = 60000
    """Final delay for T1 measurement [ns]."""
    t1_delay_step: int = 400
    """Step for T1 delay sweep [ns]."""
    ramsey_delay_start: int = 80
    """Initial delay for Ramsey measurement [ns]."""
    ramsey_delay_end: int = 60000
    """Final delay for Ramsey measurement [ns]."""
    ramsey_delay_step: int = 400
    """Step for Ramsey delay sweep [ns]."""
    ramsey_detuning: float = 0.0
    """Frequency detuning for Ramsey [Hz]."""
    nshots: int = 800
    """Number of measurement shots."""
    relaxation_time: Optional[float] = None
    """Wait time between measurements [ns]. If None, uses platform default."""


@dataclass
class InterleavedDecoherenceResults(Results):
    """Interleaved decoherence protocol results.
    
    Attributes:
        t1: T1 relaxation time for each qubit [ns].
        t2_star: T2* dephasing time for each qubit [ns].
        ramsey_frequency: Measured frequency from Ramsey for each qubit [Hz].
        fitted_parameters: Raw fitting parameters for each qubit and experiment.
    """
    
    t1: Dict[QubitId, Tuple[float, float]] = field(default_factory=dict)
    """T1 relaxation time for each qubit [ns] with error."""
    t2_star: Dict[QubitId, Tuple[float, float]] = field(default_factory=dict)
    """T2* dephasing time for each qubit [ns] with error."""
    ramsey_frequency: Dict[QubitId, Tuple[float, float]] = field(default_factory=dict)
    """Measured frequency from Ramsey for each qubit [Hz] with error."""
    qubit_detuning: Dict[QubitId, Tuple[float, float]] = field(default_factory=dict)
    """Qubit frequency detuning from Ramsey experiment [Hz] with error."""

    fitted_parameters: Dict[QubitId, Dict[str, Dict[str, float]]] = field(
        default_factory=dict
    )
    """Raw fitting parameters for each qubit and experiment."""


@dataclass
class InterleavedDecoherenceData(Data):
    """Interleaved decoherence acquisition outputs.
    
    Attributes:
        data: Raw measurement data for each qubit.
        delays: Dictionary mapping experiment types to delay arrays.
        result_map: Maps acquisition IDs to qubit and experiment information.
    """
    
    data: Dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired for each qubit."""
    delays: Dict[str, npt.NDArray] = field(default_factory=dict)
    """Delay values for each experiment type."""
    result_map: Dict[str, Dict[str, Union[QubitId, str]]] = field(default_factory=dict)
    """Maps acquisition IDs to qubit and experiment info."""
    ramsey_detuning: float = 0.0
    """Ramsey detuning used in acquisition [Hz]."""



def _acquisition(
    params: InterleavedDecoherenceParameters,
    platform: Platform,
    targets: List[QubitId],
) -> InterleavedDecoherenceData:
    """Data acquisition for interleaved decoherence protocol.
    
    Interleaves T1 and Ramsey measurements over given delays for multiple qubits.
    
    Args:
        params: Protocol parameters.
        platform: Qibolab platform.
        targets: List of qubit IDs to measure.
        
    Returns:
        InterleavedDecoherenceData with raw acquisition results.
    """
    relaxation_time = params.relaxation_time
    if relaxation_time is None:
        relaxation_time = platform.settings.relaxation_time  # in ns
    
    # Dictionary for pulses that have sweepers for each combination of qubit and test
    delay_pulses = {(q, "t1"): Delay(duration=80) for q in targets}
    delay_pulses.update({(q, "ramsey"): Delay(duration=80) for q in targets})
    rx90_phase_pulses = {}

    # Separate alignment delays for non-first qubits (swept in sync with the previous qubit's delays)
    align_t1_delays = {}    # {q: Delay} one per qubit that needs alignment
    align_ramsey_delays = {}  # {q: Delay}

    prev_durations = None  # pulse durations of the previous qubit, for alignment
    seq = PulseSequence()
    result_map = {}
    
    for q in targets:
        print(f"Preparing sequence for qubit {q}...")
        natives: SingleQubitNatives = platform.natives.single_qubit[q]
        qd_channel, rx180_pulse = natives.RX()[0]
        ro_channel = platform.qubits[q].acquisition

        _, ro_pulse_t1 = natives.MZ()[0]
        _, ro_pulse_ramsey = natives.MZ()[0]

        rx90 = natives.R(theta=np.pi / 2)
        rx90_with_phase = natives.R(theta=np.pi / 2, phi=0)
        rx90_phase_pulses[q] = rx90_with_phase[0][1]

        if prev_durations is not None:
            # Align this qubit's channels to start after the previous qubit's full sequence
            # (T1 + relaxation + Ramsey) to avoid simultaneous execution and crosstalk.
            # Separate Delay objects are added to the sweepers so they track the previous
            # qubit's swept delays automatically.
            align_t1_delays[q] = Delay(duration=80)
            align_ramsey_delays[q] = Delay(duration=80)

            prev_rx180  = prev_durations['rx180']
            prev_ro_t1  = prev_durations['ro_t1']
            prev_rx90   = prev_durations['rx90']
            prev_ro_ram = prev_durations['ro_ramsey']

            # Drive channel: mirrors the previous qubit's drive timeline
            seq.append((qd_channel, Delay(duration=prev_rx180)))
            seq.append((qd_channel, align_t1_delays[q]))
            seq.append((qd_channel, Delay(duration=prev_ro_t1 + relaxation_time + prev_rx90)))
            seq.append((qd_channel, align_ramsey_delays[q]))
            seq.append((qd_channel, Delay(duration=prev_rx90 + prev_ro_ram)))

            # Acquisition channel: mirrors the previous qubit's acquisition timeline
            seq.append((ro_channel, Delay(duration=prev_rx180)))
            seq.append((ro_channel, align_t1_delays[q]))
            seq.append((ro_channel, Delay(duration=prev_ro_t1 + relaxation_time + prev_rx90 * 2)))
            seq.append((ro_channel, align_ramsey_delays[q]))
            seq.append((ro_channel, Delay(duration=prev_ro_ram)))

        # Sequence for T1
        seq.append((qd_channel, rx180_pulse))
        seq.append((qd_channel, delay_pulses[(q, "t1")]))
        seq.append((qd_channel, Delay(duration=ro_pulse_t1.duration)))
        
        seq.append((ro_channel, Delay(duration=rx180_pulse.duration)))
        seq.append((ro_channel, delay_pulses[(q, "t1")]))
        seq.append((ro_channel, ro_pulse_t1))
        result_map[seq[-1][1].acquisition.id_] = {"qubit": q, "experiment": "t1"}

        # Relaxation time
        seq.append((ro_channel, Delay(duration=relaxation_time)))
        seq.append((qd_channel, Delay(duration=relaxation_time)))

        # Sequence for Ramsey
        seq += rx90
        seq.append((qd_channel, delay_pulses[(q, "ramsey")]))
        seq += rx90_with_phase
        seq.append((qd_channel, Delay(duration=ro_pulse_ramsey.duration)))

        seq.append((ro_channel, Delay(duration=rx90.duration * 2)))
        seq.append((ro_channel, delay_pulses[(q, "ramsey")]))
        seq.append((ro_channel, ro_pulse_ramsey))
        result_map[seq[-1][1].acquisition.id_] = {"qubit": q, "experiment": "ramsey"}

        prev_durations = {
            'rx180':    rx180_pulse.duration,
            'ro_t1':    ro_pulse_t1.duration,
            'rx90':     rx90.duration,
            'ro_ramsey': ro_pulse_ramsey.duration,
        }

    # For Debugging: Save generated sequence to a file
    # TODO: Throw this away for qibocal deployment
    print(f"Saved to generated_sequence.py")
    with open(f'generated_sequence.py', 'w') as f:
        f.write('from qibolab import Delay, Pulse, Readout, Delay, Acquisition\n')
        f.write('from qibolab import Drag, Rectangular, Gaussian, GaussianSquare\n')
        f.write('from uuid import UUID\n\n')
        f.write('sequence =  ')
        rich.print(seq, file=f)

    # Prepare sweepers
    t1_delays = np.arange(
        params.t1_delay_start, 
        params.t1_delay_end + params.t1_delay_step, 
        params.t1_delay_step
    )
    ramsey_delays = np.arange(
        params.ramsey_delay_start,
        params.ramsey_delay_end + params.ramsey_delay_step,
        params.ramsey_delay_step,
    )
    phase_values = 2 * np.pi * params.ramsey_detuning * ramsey_delays

    sweepers = {
        "t1": Sweeper(
            parameter=Parameter.duration,
            values=t1_delays,
            pulses=[delay_pulses[(q, "t1")] for q in targets]
                 + [align_t1_delays[q] for q in targets[1:]],
        ),
        "ramsey": Sweeper(
            parameter=Parameter.duration,
            values=ramsey_delays,
            pulses=[delay_pulses[(q, "ramsey")] for q in targets]
                 + [align_ramsey_delays[q] for q in targets[1:]],
        ),
        "ramsey_phase": Sweeper(
            parameter=Parameter.relative_phase,
            values=phase_values,
            pulses=[rx90_phase_pulses[q] for q in targets],
        ),
    }

    

    # Execute the sequence
    results = platform.execute(
        sequences=[seq],
        sweepers=[list(sweepers.values())],
        nshots=params.nshots,
        averaging_mode=AveragingMode.SINGLESHOT,
    )

    # Store results in data structure
    data = InterleavedDecoherenceData(
        delays={"t1": t1_delays, "ramsey": ramsey_delays},
        result_map=result_map,
        ramsey_detuning=params.ramsey_detuning,
    )

    # Organize results by qubit and experiment
    for uid, result in results.items():
        info = result_map[uid]
        qubit = info["qubit"]
        experiment = info["experiment"]
        delays_array = data.delays[experiment]
        
        # Calculate probabilities
        probs = result.sum(axis=0) / params.nshots
        errors = np.sqrt(probs * (1 - probs) / params.nshots)
        
        # Store in structured array
        qubit_data = np.zeros(len(delays_array), dtype=InterleavedDecoherenceType)
        qubit_data["delay"] = delays_array
        qubit_data["prob"] = probs
        qubit_data["error"] = errors
        qubit_data["experiment"] = experiment
        
        if qubit not in data.data:
            data.data[qubit] = []
        data.data[qubit].append(qubit_data)
    
    # Concatenate data for each qubit
    for qubit in data.data:
        data.data[qubit] = np.concatenate(data.data[qubit])

    return data


def _exponential_decay(t, a, b, t_decay):
    """Exponential decay function for T1 fitting."""
    return a + b * np.exp(-t / t_decay)


def _damped_oscillation(t, a, b, t_decay, freq, phase):
    """Damped oscillation function for Ramsey fitting."""
    return a + b * np.exp(-t / t_decay) * np.cos(2 * np.pi * freq * t + phase)


def _fit(data: InterleavedDecoherenceData) -> InterleavedDecoherenceResults:
    """Fit T1 and Ramsey data to extract coherence times.
    
    Args:
        data: Acquired data from the protocol.
        
    Returns:
        InterleavedDecoherenceResults with fitted parameters.
    """
    from scipy.optimize import curve_fit
    
    results = InterleavedDecoherenceResults()
    
    for qubit in data.data:
        qubit_data = data.data[qubit]
        
        # Separate T1 and Ramsey data
        t1_mask = qubit_data["experiment"] == "t1"
        ramsey_mask = qubit_data["experiment"] == "ramsey"
        
        t1_data = qubit_data[t1_mask]
        ramsey_data = qubit_data[ramsey_mask]
        
        # Initialize fitted parameters dict for this qubit
        results.fitted_parameters[qubit] = {}
        
        # Fit T1 data
        try:
            t1_delays = t1_data["delay"]
            t1_probs = t1_data["prob"]
            
            # Initial guess: a=0, b=1, t1=mean of delays
            p0_t1 = [0.0, 1.0, np.mean(t1_delays)]
            popt_t1, pcov_t1 = curve_fit(
                _exponential_decay,
                t1_delays,
                t1_probs,
                p0=p0_t1,
                maxfev=10000,
            )
            
            t1_value = abs(popt_t1[2])
            t1_error = np.sqrt(np.diag(pcov_t1)[2]) if pcov_t1 is not None else 0.0
            
            results.t1[qubit] = (t1_value, t1_error)
            results.fitted_parameters[qubit]["t1"] = {
                "a": popt_t1[0],
                "b": popt_t1[1],
                "t1": popt_t1[2],
            }
        except Exception as e:
            print(f"T1 fitting failed for qubit {qubit}: {e}")
            results.t1[qubit] = (0.0, 0.0)
            results.fitted_parameters[qubit]["t1"] = {}
        
        # Fit Ramsey data
        try:
            ramsey_delays = ramsey_data["delay"]
            ramsey_probs = ramsey_data["prob"]
            
            # Initial guess for damped oscillation
            # Estimate frequency from detuning
            freq_guess = data.ramsey_detuning if data.ramsey_detuning != 0 else 0
            p0_ramsey = [0.5, 0.5, np.mean(ramsey_delays), freq_guess, 0.0]
            
            popt_ramsey, pcov_ramsey = curve_fit(
                _damped_oscillation,
                ramsey_delays,
                ramsey_probs,
                p0=p0_ramsey,
                maxfev=10000,
            )
            
            t2_value = abs(popt_ramsey[2])
            t2_error = (
                np.sqrt(np.diag(pcov_ramsey)[2]) if pcov_ramsey is not None else 0.0
            )
            freq_value = popt_ramsey[3]*GHz # Convert from GHz to Hz
            freq_error = (
                np.sqrt(np.diag(pcov_ramsey)[3]) if pcov_ramsey is not None else 0.0
            )

            qubit_detuning = freq_value - data.ramsey_detuning*GHz  # in Hz

            results.t2_star[qubit] = (t2_value, t2_error)
            results.ramsey_frequency[qubit] = (freq_value, freq_error)
            results.qubit_detuning[qubit] = (qubit_detuning, freq_error)
            results.fitted_parameters[qubit]["ramsey"] = {
                "a": popt_ramsey[0],
                "b": popt_ramsey[1],
                "t2": popt_ramsey[2],
                "freq": popt_ramsey[3],
                "phase": popt_ramsey[4],
            }
        except Exception as e:
            print(f"Ramsey fitting failed for qubit {qubit}: {e}")
            results.t2_star[qubit] = (0.0, 0.0)
            results.ramsey_frequency[qubit] = (0.0, 0.0)
            results.fitted_parameters[qubit]["ramsey"] = {}
    
    return results


def _plot(
    data: InterleavedDecoherenceData,
    target: QubitId,
    fit: Optional[InterleavedDecoherenceResults] = None,
):
    """Plot T1 and Ramsey data for a single qubit.
    
    Args:
        data: Acquired data.
        target: Qubit to plot.
        fit: Optional fit results.
        
    Returns:
        List of plotly figures and fitting report HTML.
    """
    from plotly.subplots import make_subplots
    
    figures = []
    
    if target not in data.data:
        return figures, "No data available for this qubit."
    
    qubit_data = data.data[target]
    
    # Separate T1 and Ramsey data
    t1_mask = qubit_data["experiment"] == "t1"
    ramsey_mask = qubit_data["experiment"] == "ramsey"
    
    t1_data = qubit_data[t1_mask]
    ramsey_data = qubit_data[ramsey_mask]
    
    # Create subplot figure with 1 row and 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"T1 Measurement - Qubit {target}", f"Ramsey Measurement - Qubit {target}"),
        horizontal_spacing=0.12
    )
    
    # Add T1 data to first subplot
    fig.add_trace(
        go.Scatter(
            x=t1_data["delay"],
            y=t1_data["prob"],
            error_y=dict(type="data", array=t1_data["error"]),
            mode="markers",
            name="T1 Data",
            marker=dict(size=6),
            legendgroup="t1",
        ),
        row=1, col=1
    )
    
    if fit is not None and target in fit.t1:
        t1_delays = t1_data["delay"]
        t_fit = np.linspace(t1_delays.min(), t1_delays.max(), 200)
        params = fit.fitted_parameters[target].get("t1", {})
        if params:
            y_fit = _exponential_decay(
                t_fit, params["a"], params["b"], params["t1"]
            )
            fig.add_trace(
                go.Scatter(
                    x=t_fit,
                    y=y_fit,
                    mode="lines",
                    name=f"T1 Fit ({fit.t1[target][0]:.0f}±{fit.t1[target][1]:.0f} ns)",
                    line=dict(dash="dash"),
                    legendgroup="t1",
                ),
                row=1, col=1
            )
    
    # Add Ramsey data to second subplot
    fig.add_trace(
        go.Scatter(
            x=ramsey_data["delay"],
            y=ramsey_data["prob"],
            error_y=dict(type="data", array=ramsey_data["error"]),
            mode="markers",
            name="Ramsey Data",
            marker=dict(size=6),
            legendgroup="ramsey",
        ),
        row=1, col=2
    )
    
    if fit is not None and target in fit.t2_star:
        ramsey_delays = ramsey_data["delay"]
        t_fit = np.linspace(ramsey_delays.min(), ramsey_delays.max(), 200)
        params = fit.fitted_parameters[target].get("ramsey", {})
        if params:
            y_fit = _damped_oscillation(
                t_fit,
                params["a"],
                params["b"],
                params["t2"],
                params["freq"],
                params["phase"],
            )
            fig.add_trace(
                go.Scatter(
                    x=t_fit,
                    y=y_fit,
                    mode="lines",
                    name=f"T2* Fit ({fit.t2_star[target][0]:.0f}±{fit.t2_star[target][1]:.0f} ns)",
                    line=dict(dash="dash"),
                    legendgroup="ramsey",
                ),
                row=1, col=2
            )
        # Calculate the qubit frequency detuning using the fitted frequency and ramsey detuning
        if data.ramsey_detuning != 0:
            fitted_freq = fit.ramsey_frequency[target][0]
            freq_dettuning = (fitted_freq - data.ramsey_detuning)*1e9  # in Hz
    
    # Update axis labels
    fig.update_xaxes(title_text="Delay (ns)", row=1, col=1)
    fig.update_xaxes(title_text="Delay (ns)", row=1, col=2)
    fig.update_yaxes(title_text="|1⟩ Population", row=1, col=1)
    fig.update_yaxes(title_text="|1⟩ Population", row=1, col=2)
    
    # Update overall layout
    fig.update_layout(
        showlegend=True,
        height=500,
        title_text=f"Interleaved Decoherence Measurements - Qubit {target}",
    )
    
    figures.append(fig)
    
    # Generate fitting report
    fitting_report = "<h3>Fitting Results</h3>"
    if fit is not None:
        fitting_report += f"<p><b>Qubit {target}</b></p>"
        if target in fit.t1:
            fitting_report += f"<p>T1: {fit.t1[target][0]:.0f} ± {fit.t1[target][1]:.0f} ns</p>"
        if target in fit.t2_star:
            fitting_report += f"<p>T2*: {fit.t2_star[target][0]:.0f} ± {fit.t2_star[target][1]:.0f} ns</p>"
        if target in fit.ramsey_frequency:
            fitting_report += f"<p>Ramsey Frequency: {fit.ramsey_frequency[target][0]:.3e} ± {fit.ramsey_frequency[target][1]:.3e} Hz</p>"
    
    return figures, fitting_report

# Create the Routine object if qibocal is available

interleaved_decoherence = Routine(_acquisition, _fit, _plot)

# Standalone execution

# TODO: Remove this for the qibocal version and use the Routine object instead.
def main():
    parser = argparse.ArgumentParser(
        description="Long qubit spectroscopy with LO adjustment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python interleaved_coherence.py qpu154 0 --t1_delay_start 80 --t1_delay_end 64080 --t1_delay_step 160 \\
    --ramsey_delay_start 80 --ramsey_delay_end 16080 --ramsey_delay_step 40 --ramsey_detuning 0.000375 --nshots 400

        """)
    
    parser.add_argument("platform", type=str, help="Platform name")
    parser.add_argument("qubits", type=str, nargs='+', help="Target qubit(s)")
    parser.add_argument("--t1_delay_start", type=int, default=80,
                        help="Initial delay for T1 measurement [ns], default: 80")
    parser.add_argument("--t1_delay_end", type=int, default=64080,
                        help="Final delay for T1 measurement [ns], default: 64080")
    parser.add_argument("--t1_delay_step", type=int, default=320,
                        help="Step for T1 delay sweep [ns], default: 320")
    parser.add_argument("--ramsey_delay_start", type=int, default=80,
                        help="Initial delay for Ramsey measurement [ns], default: 80")
    parser.add_argument("--ramsey_delay_end", type=int, default=16080,
                        help="Final delay for Ramsey measurement [ns], default: 16080")
    parser.add_argument("--ramsey_delay_step", type=int, default=80,
                        help="Step for Ramsey delay sweep [ns], default: 80")
    parser.add_argument("--ramsey_detuning", type=float, default=6/16000,
                        help="Frequency detuning for Ramsey [Hz], default: 6/16000")
    parser.add_argument("--nshots", type=int, default=500,
                        help="Number of measurement shots, default: 500")
    
    args = parser.parse_args()

    from qibolab import create_platform
    import time
    from datetime import timezone

     # Load the platform
    platform_name = args.platform
    # username = os.environ["USER"]
    # os.environ["QIBOLAB_PLATFORMS"] = f"/home/users/{username}/qibolab_platforms_qrc"

    # Define Parameters acquisition
    params = InterleavedDecoherenceParameters(
        t1_delay_start=args.t1_delay_start,
        t1_delay_end=args.t1_delay_end,
        t1_delay_step=args.t1_delay_step,
        ramsey_delay_start=args.ramsey_delay_start,
        ramsey_delay_end=args.ramsey_delay_end,
        ramsey_delay_step=args.ramsey_delay_step,
        ramsey_detuning=args.ramsey_detuning,
        nshots=args.nshots,
    )

    
    # Execute acquisition
    platform: Platform = create_platform(platform_name)
    test_sets = [args.qubits] # All qubits together

    set_time: dict = {} # To store acquisition time for each qubit set
    data = []
    try:
        for i, qubits in enumerate(test_sets):
            print(f"Starting acquisition for set: [{qubits}]...")
            test_time = time.time()
            for qubit in qubits:
                set_time[qubit] = test_time
            platform: Platform = create_platform(platform_name) # to run multiple sets sequentially we need to create a new platform each time
            platform.connect()
            if i == 0:
                data = _acquisition(params, platform, qubits)
            else:
                data_ = _acquisition(params, platform, qubits)
                data.data.update(data_.data)  # Combine data from both runs
                data.result_map.update(data_.result_map)
            platform.disconnect()
    finally:
        end_time = time.time()
    
    # Fit the data
    results = _fit(data)
    
    # Print results and save to csv file

    for qubit in np.array(test_sets).flatten():
        print(f"Results for Qubit {qubit}:")
        print(f"Acquisition Time: {end_time - set_time[qubit]:.2f} seconds")
        
        t1_val, t1_err = results.t1[qubit]
        print(f"Qubit {qubit} - T1: {t1_val:.0f} ± {t1_err:.0f} ns")
    
        t2_val, t2_err = results.t2_star[qubit]
        print(f"Qubit {qubit} - T2*: {t2_val:.0f} ± {t2_err:.0f} ns")
    
        freq_val, freq_err = results.ramsey_frequency[qubit]
        print(f"Qubit {qubit} - Ramsey Freq: {freq_val:.3e} ± {freq_err:.3e} Hz")
    
        det_val, det_err = results.qubit_detuning[qubit]
        print(f"Qubit {qubit} - Qubit Detuning: {det_val:.3e} ± {det_err:.3e} Hz")

        # Save results to CSV
        
        with open(f"monitor_{platform_name}.csv", mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(["Timestamp", "Qubit", "T1", "T2*", "Ramsey Frequency", "Qubit Detuning"])
            
            # Save time as UTC+4 (Abu Dhabi)
            tz = timezone(timedelta(hours=4))
            time_ = time.localtime(set_time[qubit] + tz.utcoffset(None).total_seconds())

            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time_),
                             qubit,
                             f"{results.t1[qubit][0]:.0f} ± {results.t1[qubit][1]:.0f}",
                             f"{results.t2_star[qubit][0]:.0f} ± {results.t2_star[qubit][1]:.0f}",
                             f"{results.ramsey_frequency[qubit][0]:.3e} ± {results.ramsey_frequency[qubit][1]:.3e}",
                             f"{results.qubit_detuning[qubit][0]:.3e} ± {results.qubit_detuning[qubit][1]:.3e}"]
                             )
            
    # Plot results
    for qubit in range(5):
        figs, report = _plot(data, qubit, results)
        for i, fig in enumerate(figs):
            fig.write_html(f"interleaved_decoherence_qubit{qubit}_{i}.html")

if __name__ == "__main__":
    main()