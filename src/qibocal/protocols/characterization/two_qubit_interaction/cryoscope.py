"""Cryoscope experiment for two qubit gates, cryscope plot."""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import FluxPulse, PulseSequence, Rectangular
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from scipy.optimize import curve_fit

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.config import log


@dataclass
class CryscopeParameters(Parameters):
    """Cryoscope runcard inputs."""

    flux_pulse_amplitude: float
    flux_pulse_duration_start: int
    """Duration minimum."""
    flux_pulse_duration_end: int
    """Duration maximum."""
    flux_pulse_duration_step: int
    """Duration step."""
    flux_pulse_amplitude_start: float
    """Amplitude minimum."""
    flux_pulse_amplitude_end: float
    """Amplitude maximum."""
    flux_pulse_amplitude_step: float
    """Amplitude step."""
    delay_before_flux: int
    delay_after_flux: int
    flux_pulse_shapes: Optional[dict] = None
    nshots: Optional[int] = None
    """Number of shots per point."""
    relaxation_time: Optional[int] = None


@dataclass
class CryscopeResults(Results):
    """Cryoscope outputs."""

    # FIXME: update runcard accordingly
    filter_coefs: list[float]
    """FIR IIR coefficients to predistor a flux pulse"""


CryscopeType = np.dtype(
    [
        ("statistical_freq", np.float64),
        ("flux_pulse_duration", np.float64),
        ("flux_pulse_amplitude", np.float64),
    ]
)
"""Custom dtype for cryscope."""


@dataclass
class CryscopeData(Data):
    """Cryscope acquisition outputs."""

    data: dict[tuple[QubitId, str], npt.NDArray[CryscopeType]] = field(
        default_factory=dict
    )

    def register_qubit(
        self,
        qubit,
        component,
        flux_pulse_duration,
        flux_pulse_amplitude,
        statistical_freq,
    ):
        """Store output for single qubit."""
        size = len(flux_pulse_duration) * len(flux_pulse_amplitude)
        _flux_pulse_duration, _flux_pulse_amplitude = np.meshgrid(
            flux_pulse_duration, flux_pulse_amplitude
        )
        ar = np.empty(size, dtype=CryscopeType)
        ar["flux_pulse_duration"] = _flux_pulse_duration.ravel()
        ar["flux_pulse_amplitude"] = _flux_pulse_amplitude.ravel()
        ar["statistical_freq"] = statistical_freq.ravel()
        self.data[qubit, component] = np.rec.array(ar)


def _aquisition(
    params: CryscopeParameters,
    platform: Platform,
    qubits: Qubits,
) -> CryscopeData:
    r"""
    The cryoscope is one of the experiments required to characterise 2-qubit gates. Its aim is to measure
    the distorsions suffered by a flux pulse on its way from the control instrument to the qubit, so that they can later
    be corrected. Correcting these distorsions allows to reliably control the detuning of a qubit to a desired
    point where it interacts with a second qubit.
    In order to meassure the distorsions, the control qubit (the one with the highest frequency) is first prepared
    on a |+> state, a flux pulse of varying duration and amplitude is applied, finally the phase accumulated around
    the Z axis is meassured.
    For a given flux pulse amplitude, the derivative of the phase with respect to the pulse duration gives the instant
    detunning.
    Although the shape of the flux pulse sent is square, the impedance of the line from the instrument to the qubit,
    distort the pulse, resulting in a detuning with transients at the beginngin and end of the pulse.
    For every flux pulse amplitude, the stable detuning (after excluding those transients) can be measured.
    Those points are then fitted and this non-linear relation is used to determine the instant amplitude of the distorted
    pulse, as it is seen by the qubit, from the instant detunings measured.
    Once the distorted pulse is reconstructed, a convolution can be calculated such that pre-applied to the flux pulse,
    counters the distorsions.
    The type of convolutions chose are those that can be implemented in real time by control instruments: FIR and IIR
    filters.
    One filter may not be enough to counter all distorsions, so the cryoscope experiment can be iterated multiple times
    to determine multiple filters.

    https://arxiv.org/abs/1907.04818

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (list): List of target qubits to perform the action
        flux_pulse_duration_start (int): Initial flux pulse duration
        flux_pulse_duration_end (int): Maximum flux pulse duration
        flux_pulse_duration_step (int): Scan range step for the flux pulse duration
        flux_pulse_amplitude_start (int): Initial flux pulse amplitude
        flux_pulse_amplitude_end (int): Maximum flux pulse amplitude
        flux_pulse_amplitude_step (int): Scan range step for the flux pulse amplitude
        delay_before_flux (int): A time delay in ns before the flux pulse
        delay_before_readout (int): A time delay in ns after the flux pulse
        flux_pulse_shapes (dict(PulseShape)): A dictionary of qubit_ids: PulseShape objects to be used for each qubit flux pulse.
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Returns:
        - A DataUnits object with the raw data with the following keys
            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **prob[dimensionless]**: Statistical relative frequency of measuring state |1>
            - **flux_pulse_amplitude[dimensionless]**: Flux pulse amplitude
            - **flux_pulse_duration[ns]**: Flux pulse duration in ns
            - **component**: The component being measured [MX, MY]
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages

    """

    # 1) from state |0> apply Ry(pi/2) to state |+>,
    # 2) apply a flux pulse of variable duration,
    # 3) wait for the transient of the flux pulse to disappear
    # 3) measure in the X and Y axis
    #   MX = Ry(pi/2) - delay before - (flux)(t) - delay after - Ry(pi/2)  - MZ
    #   MY = Ry(pi/2) - delay before - (flux)(t) - delay after - Rx(-pi/2) - MZ
    #   MZ = Ry(pi/2) - delay before - (flux)(t) - delay after -           - MZ
    #   M0 =                                                               - MZ
    #   M1 = Ry(pi) -                                                      - MZ

    # The flux pulse detunes the qubit and results in a rotation around the Z axis = atan(MY/MX)

    flux_pulse_amplitude: float = params.flux_pulse_amplitude
    flux_pulse_duration_start: int = params.flux_pulse_duration_start
    flux_pulse_duration_end: int = params.flux_pulse_duration_end
    flux_pulse_duration_step: int = params.flux_pulse_duration_step
    flux_pulse_amplitude_start: float = params.flux_pulse_amplitude_start
    flux_pulse_amplitude_end: float = params.flux_pulse_amplitude_end
    flux_pulse_amplitude_step: float = params.flux_pulse_amplitude_step
    delay_before_flux: int = params.delay_before_flux
    delay_after_flux: int = params.delay_after_flux
    flux_pulse_shapes: dict = params.flux_pulse_shapes
    nshots: int = params.nshots
    relaxation_time = params.relaxation_time

    # define the sequences of pulses to be executed
    MX_seq = PulseSequence()
    MY_seq = PulseSequence()
    MZ_seq = PulseSequence()
    M0_seq = PulseSequence()
    M1_seq = PulseSequence()

    initial_RY90_pulses = {}
    flux_pulses = {}
    RX90_pulses = {}
    RY90_pulses = {}
    MZ_ro_pulses = {}
    for qubit in qubits:
        # start at |+> by rotating Ry(pi/2)
        initial_RY90_pulses[qubit] = platform.create_RX90_pulse(
            qubit, start=0, relative_phase=np.pi / 2
        )

        if flux_pulse_shapes and len(flux_pulse_shapes) == len(qubits):
            flux_pulse_shape = eval(flux_pulse_shapes[qubit])
        else:
            flux_pulse_shape = Rectangular()

        # wait before applying flux pulse (delay_before_flux)

        # apply a detuning flux pulse
        flux_pulses[qubit] = FluxPulse(
            start=initial_RY90_pulses[qubit].se_finish + delay_before_flux,
            duration=flux_pulse_duration_end,  # sweep to produce oscillations
            amplitude=flux_pulse_amplitude,  # fix for each run
            shape=flux_pulse_shape,
            channel=platform.qubits[qubit].flux.name,
            qubit=platform.qubits[qubit].name,
        )

        # wait after applying flux pulse (delay_after_flux)

        # rotate around the X asis Rx(-pi/2) to meassure Y component
        RX90_pulses[qubit] = platform.create_RX90_pulse(
            qubit,
            start=flux_pulses[qubit].finish + delay_after_flux,
            relative_phase=np.pi,
        )

        # rotate around the Y asis Ry(pi/2) to meassure X component
        RY90_pulses[qubit] = platform.create_RX90_pulse(
            qubit,
            start=flux_pulses[qubit].finish + delay_after_flux,
            relative_phase=np.pi / 2,
        )

        # add ro pulse at the end of each sequence
        MZ_ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX90_pulses[qubit].finish
        )

        # add pulses to the sequences
        MX_seq.add(
            initial_RY90_pulses[qubit],
            flux_pulses[qubit],
            RY90_pulses[qubit],
            MZ_ro_pulses[qubit],
        )
        MY_seq.add(
            initial_RY90_pulses[qubit],
            flux_pulses[qubit],
            RX90_pulses[qubit],
            MZ_ro_pulses[qubit],
        )
        MZ_seq.add(
            initial_RY90_pulses[qubit],
            flux_pulses[qubit],
            MZ_ro_pulses[qubit],
        )
        M0_seq.add(
            MZ_ro_pulses[qubit],
        )
        M1_seq.add(
            platform.create_RX_pulse(qubit, start=0, relative_phase=np.pi / 2),
            MZ_ro_pulses[qubit],
        )

        # DEBUG: Plot Cryoscope Sequences
        # MX_seq.plot("MX_seq")
        # MY_seq.plot("MY_seq")
        # MZ_seq.plot("MZ_seq")
        # M0_seq.plot("M0_seq")
        # M1_seq.plot("M1_seq")

    MX_tag = "MX"
    MY_tag = "MY"
    MZ_tag = "MZ"
    M0_tag = "M0"
    M1_tag = "M1"

    # define the parameters to sweep and their range:
    # flux pulse amplitude
    # flux pulse duration

    flux_pulse_amplitude_range = np.arange(
        flux_pulse_amplitude_start, flux_pulse_amplitude_end, flux_pulse_amplitude_step
    )
    amplitude_sweeper = Sweeper(
        Parameter.amplitude,
        flux_pulse_amplitude_range,
        pulses=[flux_pulses[qubit] for qubit in qubits],
        # type=SweeperType.ABSOLUTE,
        type=SweeperType.OFFSET,
    )
    flux_pulse_duration_range = np.arange(
        flux_pulse_duration_start, flux_pulse_duration_end, flux_pulse_duration_step
    )
    duration_sweeper = Sweeper(
        Parameter.duration,
        flux_pulse_duration_range,
        pulses=[flux_pulses[qubit] for qubit in qubits],
        type=SweeperType.ABSOLUTE,
    )

    # create a CryscopeData object to store the results,
    data = CryscopeData()
    for qubit in qubits:
        # execute the pulse sequences
        for sequence, tag in [
            (MX_seq, MX_tag),
            (MY_seq, MY_tag),
            (MZ_seq, MZ_tag),
            (M0_seq, M0_tag),
            (M1_seq, M1_tag),
        ]:
            results = platform.sweep(
                sequence,
                ExecutionParameters(
                    nshots=params.nshots,
                    acquisition_type=AcquisitionType.DISCRIMINATION,
                    averaging_mode=AveragingMode.CYCLIC,
                ),
                duration_sweeper,
                amplitude_sweeper,
            )
            # store the results
            data.register_qubit(
                qubit,
                component=tag,
                flux_pulse_duration=flux_pulse_duration_range,
                flux_pulse_amplitude=flux_pulse_amplitude_range,
                statistical_freq=results[MZ_ro_pulses[qubit].serial].probability,
            )


def _plot(data: CryscopeData, fit: CryscopeResults, qubits):
    """Plot the experiment result for a single pair."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("low", "high"))
    states = ["low", "high"]
    # Plot data
    colouraxis = ["coloraxis", "coloraxis2"]

    fit_report = ""
    for state, qubit in zip(states, qubits):
        index = (qubits[0], qubits[1], qubit)
        ordered_index = index if index in data.data else (index[1], index[0], index[2])
        fig.add_trace(
            go.Heatmap(
                x=data[ordered_index].length,
                y=data[ordered_index].amp,
                z=data[ordered_index].prob,
                name=f"Qubit {qubit} |{state}>",
                coloraxis=colouraxis[states.index(state)],
            ),
            row=1,
            col=states.index(state) + 1,
        )

        fit_report += f"q{qubit} - {state} frequency| "
        fit_report += f"Period of oscillation: {fit.period[str(ordered_index)]} ns<br>"

        fig.update_layout(
            title=f"Qubits {qubits[0]}-{qubits[1]} swap frequency",
            xaxis_title="Duration [ns]",
            yaxis_title="Amplitude [dimensionless]",
            legend_title="States",
        )
        fig.update_layout(
            coloraxis={"colorscale": "Oryel", "colorbar": {"x": -0.15}},
            coloraxis2={"colorscale": "Darkmint", "colorbar": {"x": 1.15}},
        )
    return [fig], fit_report


def fit_function(x, p0, p1, p2, p3):
    """Sinusoidal fit function."""
    return p0 + p1 * np.sin(2 * np.pi * p2 * x + p3)


def _fit(data: CryscopeData):
    pairs = data.pairs
    results = {}
    for pair in pairs:
        for qubit in pair:
            qubit_data = data[pair[0], pair[1], qubit]
            fft_freqs = []
            for amp in np.unique(qubit_data.amp):
                probability = qubit_data[qubit_data.amp == amp].prob
                fft_freqs.append(max(np.abs(np.fft.fft(probability))))

            min_idx = np.argmin(fft_freqs)
            amp = np.unique(qubit_data.amp)[min_idx]
            duration = qubit_data[qubit_data.amp == amp].length
            probability = qubit_data[qubit_data.amp == amp].amp
            guesses = [np.mean(probability), 1, np.min(fft_freqs), 0]
            # bounds = []
            # TODO maybe normalize
            try:
                popt, _ = curve_fit(
                    fit_function, duration, probability, p0=guesses, maxfev=10000
                )

                results[str((pair[0], pair[1], qubit))] = np.abs(1 / popt[2])

            except:
                log.warning("cryscope fit: the fitting was not succesful")

                results[str((pair, qubit))] = 0

    return CryscopeResults(results)


cryoscope = Routine(_aquisition, _fit, _plot)
"""Cryscope routine."""


# @plot("Flux pulse timing", plots.flux_pulse_timing)
# def flux_pulse_timing(
#     platform: AbstractPlatform,
#     qubits: list,
#     flux_pulse_amplitude_start,
#     flux_pulse_amplitude_end,
#     flux_pulse_amplitude_step,
#     flux_pulse_start_start,
#     flux_pulse_start_end,
#     flux_pulse_start_step,
#     flux_pulse_duration,
#     time_window,
#     nshots=1024,
#     relaxation_time=None,
#     software_averages=1,
# ):
#     r"""
#     The flux pulse timing experiment aims to determine the relative time of flight between
#     the drive lines and the flux lines, as well as the time required for the transient response to
#     disappear when the flux pulse stops. This is later required for the correct scheduling of
#     the flux pulses in the implementation of 2q gates.

#     Args:
#         platform (AbstractPlatform): Qibolab platform object
#         qubits (list): List of target qubits to perform the action
#         flux_pulse_amplitude_start (int): Initial flux pulse amplitude
#         flux_pulse_amplitude_end (int): Maximum flux pulse amplitude
#         flux_pulse_amplitude_step (int): Scan range step for the flux pulse amplitude
#         flux_pulse_start_start (int): The earliest start time of the flux pluse, in ns,
#             relative to the start of the first pulse of the sequence Ry(pi/2)
#         flux_pulse_start_end (int): The earliest start time of the flux pluse, in ns
#         flux_pulse_start_step (int): Scan range step for the flux pulse start
#         flux_pulse_duration (int): The duration of the flux pulse
#         time_window (int): The time window in ns between the end of the first pulse of the sequence Ry(pi/2)
#             and the beginning of the second pulse of the sequence Ry(pi/2)
#         software_averages (int): Number of executions of the routine for averaging results
#         points (int): Save data results in a file every number of points

#     Returns:
#         - A DataUnits object with the raw data with the following keys
#             - **MSR[V]**: Resonator signal voltage mesurement in volts
#             - **i[V]**: Resonator signal voltage mesurement for the component I in volts
#             - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
#             - **phase[rad]**: Resonator signal phase mesurement in radians
#             - **prob[dimensionless]**: Statistical relative frequency of measuring state |1>
#             - **flux_pulse_amplitude[dimensionless]**: Flux pulse amplitude
#             - **flux_pulse_start[ns]**:The absolut start time of the flux pulse, in ns
#             - **qubit**: The qubit being tested
#             - **iteration**: The iteration number of the many determined by software_averages

#         - A Data object with the following parameters
#             - **flux_pulse_duration**: The duration of the flux pulse
#             - **time_window**: The time window in ns between the end of the first pulse of the sequence Ry(pi/2)
#                     and the beginning of the second pulse of the sequence Ry(pi/2)
#             - **initial_RY90_pulses_finish**: The absolut finish time of the first pulse of the sequence Ry(pi/2)
#             - **qubit**: The qubit being tested

#     """

#     # the purpose of this routine is to determine the time of flight of a flux pulse and the duration
#     # of the transient at the end of the pulse

#     # 1) from state |0> apply Ry(pi/2) to state |+>,
#     # 2) apply a detunning flux pulse of fixed duration at various start times
#     # 3) after a certain time window from the end of the initial Ry(pi/2) pulse,
#     #    measure in the X axis: Ry(pi/2) - MZ

#     #   MX = Ry(pi/2) - (flux) - Ry(pi/2)  - MZ
#     # If the flux pulse of sufficient amplitude falls within the time window between the Ry(pi/2) pulses,
#     # it detunes the qubit and results in a rotation around the Z axis r so that MX = Cos(r)

#     # reload instrument settings from runcard
#     platform.reload_settings()

#     # define the sequences of pulses to be executed
#     sequence = PulseSequence()

#     initial_RY90_pulses = {}
#     flux_pulses = {}
#     RY90_pulses = {}
#     MZ_ro_pulses = {}
#     if flux_pulse_start_start < 4:
#         initial_RY90_pulses_start = 4 - flux_pulse_start_start
#     else:
#         initial_RY90_pulses_start = 0

#     for qubit in qubits:
#         # start at |+> by rotating Ry(pi/2)
#         initial_RY90_pulses[qubit] = platform.create_RX90_pulse(
#             qubit, start=initial_RY90_pulses_start, relative_phase=np.pi / 2
#         )

#         # wait time window

#         # rotate around the Y asis Ry(pi/2) to meassure X component
#         RY90_pulses[qubit] = platform.create_RX90_pulse(
#             qubit,
#             start=initial_RY90_pulses[qubit].finish + time_window,
#             relative_phase=np.pi / 2,
#         )

#         # add ro pulse at the end of each sequence
#         MZ_ro_pulses[qubit] = platform.create_qubit_readout_pulse(
#             qubit, start=RY90_pulses[qubit].finish
#         )

#         # apply a detuning flux pulse around the time window
#         flux_pulses[qubit] = FluxPulse(
#             start=initial_RY90_pulses_start + flux_pulse_start_start,
#             duration=flux_pulse_duration,
#             amplitude=flux_pulse_amplitude_start,  # fix for each run
#             shape=Rectangular(),
#             # relative_phase=0,
#             channel=qubits[qubit].flux.name,
#             qubit=qubit,
#         )

#         # add pulses to the sequences
#         sequence.add(
#             initial_RY90_pulses[qubit],
#             flux_pulses[qubit],
#             RY90_pulses[qubit],
#             MZ_ro_pulses[qubit],
#         )

#     # define the parameters to sweep and their range:
#     # flux pulse amplitude
#     # flux pulse duration

#     flux_pulse_amplitude_range = np.arange(
#         flux_pulse_amplitude_start, flux_pulse_amplitude_end, flux_pulse_amplitude_step
#     )
#     flux_pulse_amplitude_sweeper = Sweeper(
#         Parameter.amplitude,
#         flux_pulse_amplitude_range,
#         pulses=[flux_pulses[qubit] for qubit in qubits],
#     )
#     flux_pulse_start_range = np.arange(
#         flux_pulse_start_start, flux_pulse_start_end, flux_pulse_start_step
#     )
#     if flux_pulse_start_start >= 4:
#         flux_pulse_start_sweeper = Sweeper(
#             Parameter.start,
#             flux_pulse_start_range,
#             pulses=[flux_pulses[qubit] for qubit in qubits],
#         )
#     else:
#         flux_pulse_start_sweeper = Sweeper(
#             Parameter.start,
#             np.arange(
#                 4,
#                 flux_pulse_start_end + 4 - flux_pulse_start_start,
#                 flux_pulse_start_step,
#             ),
#             pulses=[flux_pulses[qubit] for qubit in qubits],
#         )

#     # create a DataUnits object to store the results,
#     # DataUnits stores by default MSR, phase, i, q
#     # additionally include flux pulse duration, amplitude and the probability
#     data = DataUnits(
#         name=f"data",
#         quantities={
#             "flux_pulse_amplitude": "dimensionless",
#             "flux_pulse_start": "ns",
#             "prob": "dimensionless",
#         },
#         options=["qubit", "iteration"],
#     )

#     parameters = Data(
#         name=f"parameters",
#         quantities=[
#             "flux_pulse_duration",
#             "time_window",
#             "initial_RY90_pulses_finish",
#             "qubit",
#         ],
#     )
#     for qubit in qubits:
#         parameters.add(
#             {
#                 "flux_pulse_duration": flux_pulse_duration,
#                 "time_window": time_window,
#                 "initial_RY90_pulses_finish": initial_RY90_pulses[qubit].finish,
#                 "qubit": qubit,
#             }
#         )
#     yield parameters

#     start_times = np.repeat(flux_pulse_start_range, len(flux_pulse_amplitude_range))
#     amplitudes = np.tile(flux_pulse_amplitude_range, len(flux_pulse_start_range))
#     ndata = len(flux_pulse_start_range) * len(flux_pulse_amplitude_range)

#     for iteration in range(software_averages):
#         # execute the pulse sequence
#         results = platform.sweep(
#             sequence,
#             flux_pulse_start_sweeper,
#             flux_pulse_amplitude_sweeper,
#             nshots=nshots,
#             relaxation_time=relaxation_time,
#         )

#         for qubit in qubits:
#             res_temp = results[MZ_ro_pulses[qubit].serial].raw
#             res_temp.update(
#                 {
#                     "flux_pulse_start[ns]": start_times,
#                     "flux_pulse_amplitude[dimensionless]": amplitudes,
#                     "qubit": ndata * [qubit],
#                     "iteration": ndata * [iteration],
#                 }
#             )
#             data.add_data_from_dict(res_temp)
#     yield data


# @plot("cryoscope_raw", plots.cryoscope_raw)
# @plot("cryoscope_dephasing", plots.cryoscope_dephasing)
# @plot("cryoscope_dephasing_heatmap", plots.cryoscope_dephasing_heatmap)
# @plot("cryoscope_fft_peak_fitting", plots.cryoscope_fft_peak_fitting)
# @plot("cryoscope_fft", plots.cryoscope_fft)
# @plot("cryoscope_phase", plots.cryoscope_phase)
# @plot("cryoscope_phase_heatmap", plots.cryoscope_phase_heatmap)
# @plot("cryoscope_phase_unwrapped", plots.cryoscope_phase_unwrapped)
# @plot("cryoscope_phase_unwrapped_heatmap", plots.cryoscope_phase_unwrapped_heatmap)
# @plot(
#     "cryoscope_phase_amplitude_unwrapped_heatmap",
#     plots.cryoscope_phase_amplitude_unwrapped_heatmap,
# )
# @plot("cryoscope_detuning_time", plots.cryoscope_detuning_time)
# @plot("cryoscope_distorted_amplitude_time", plots.cryoscope_distorted_amplitude_time)
# @plot(
#     "cryoscope_reconstructed_amplitude_time",
#     plots.cryoscope_reconstructed_amplitude_time,
# )
# def cryoscope(
#     platform: AbstractPlatform,
#     qubits: dict,
#     flux_pulse_amplitude,
#     flux_pulse_duration_start,
#     flux_pulse_duration_end,
#     flux_pulse_duration_step,
#     flux_pulse_amplitude_start,
#     flux_pulse_amplitude_end,
#     flux_pulse_amplitude_step,
#     delay_before_flux,
#     delay_after_flux,
#     flux_pulse_shapes: Optional[dict] = None,
#     nshots=1024,
#     relaxation_time=None,
#     software_averages=1,
# ):
#     r"""
#     The cryoscope is one of the experiments required to characterise 2-qubit gates. Its aim is to measure
#     the distorsions suffered by a flux pulse on its way from the control instrument to the qubit, so that they can later
#     be corrected. Correcting these distorsions allows to reliably control the detuning of a qubit to a desired
#     point where it interacts with a second qubit.
#     In order to meassure the distorsions, the control qubit (the one with the highest frequency) is first prepared
#     on a |+> state, a flux pulse of varying duration and amplitude is applied, finally the phase accumulated around
#     the Z axis is meassured.
#     For a given flux pulse amplitude, the derivative of the phase with respect to the pulse duration gives the instant
#     detunning.
#     Although the shape of the flux pulse sent is square, the impedance of the line from the instrument to the qubit,
#     distort the pulse, resulting in a detuning with transients at the beginngin and end of the pulse.
#     For every flux pulse amplitude, the stable detuning (after excluding those transients) can be measured.
#     Those points are then fitted and this non-linear relation is used to determine the instant amplitude of the distorted
#     pulse, as it is seen by the qubit, from the instant detunings measured.
#     Once the distorted pulse is reconstructed, a convolution can be calculated such that pre-applied to the flux pulse,
#     counters the distorsions.
#     The type of convolutions chose are those that can be implemented in real time by control instruments: FIR and IIR
#     filters.
#     One filter may not be enough to counter all distorsions, so the cryoscope experiment can be iterated multiple times
#     to determine multiple filters.

#     https://arxiv.org/abs/1907.04818

#     Args:
#         platform (AbstractPlatform): Qibolab platform object
#         qubits (list): List of target qubits to perform the action
#         flux_pulse_duration_start (int): Initial flux pulse duration
#         flux_pulse_duration_end (int): Maximum flux pulse duration
#         flux_pulse_duration_step (int): Scan range step for the flux pulse duration
#         flux_pulse_amplitude_start (int): Initial flux pulse amplitude
#         flux_pulse_amplitude_end (int): Maximum flux pulse amplitude
#         flux_pulse_amplitude_step (int): Scan range step for the flux pulse amplitude
#         delay_before_flux (int): A time delay in ns before the flux pulse
#         delay_before_readout (int): A time delay in ns after the flux pulse
#         flux_pulse_shapes (dict(PulseShape)): A dictionary of qubit_ids: PulseShape objects to be used for each qubit flux pulse.
#         software_averages (int): Number of executions of the routine for averaging results
#         points (int): Save data results in a file every number of points

#     Returns:
#         - A DataUnits object with the raw data with the following keys
#             - **MSR[V]**: Resonator signal voltage mesurement in volts
#             - **i[V]**: Resonator signal voltage mesurement for the component I in volts
#             - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
#             - **phase[rad]**: Resonator signal phase mesurement in radians
#             - **prob[dimensionless]**: Statistical relative frequency of measuring state |1>
#             - **flux_pulse_amplitude[dimensionless]**: Flux pulse amplitude
#             - **flux_pulse_duration[ns]**: Flux pulse duration in ns
#             - **component**: The component being measured [MX, MY]
#             - **qubit**: The qubit being tested
#             - **iteration**: The iteration number of the many determined by software_averages

#     """

#     # 1) from state |0> apply Ry(pi/2) to state |+>,
#     # 2) apply a flux pulse of variable duration,
#     # 3) wait for the transient of the flux pulse to disappear
#     # 3) measure in the X and Y axis
#     #   MX = Ry(pi/2) - delay before - (flux)(t) - delay after - Ry(pi/2)  - MZ
#     #   MY = Ry(pi/2) - delay before - (flux)(t) - delay after - Rx(-pi/2) - MZ
#     #   MZ = Ry(pi/2) - delay before - (flux)(t) - delay after -           - MZ
#     #   M0 =                                                               - MZ
#     #   M1 = Ry(pi) -                                                      - MZ

#     # The flux pulse detunes the qubit and results in a rotation around the Z axis = atan(MY/MX)

#     # reload instrument settings from runcard
#     platform.reload_settings()

#     # define the sequences of pulses to be executed
#     MX_seq = PulseSequence()
#     MY_seq = PulseSequence()
#     MZ_seq = PulseSequence()
#     M0_seq = PulseSequence()
#     M1_seq = PulseSequence()

#     initial_RY90_pulses = {}
#     flux_pulses = {}
#     RX90_pulses = {}
#     RY90_pulses = {}
#     MZ_ro_pulses = {}
#     for qubit in qubits:
#         # start at |+> by rotating Ry(pi/2)
#         initial_RY90_pulses[qubit] = platform.create_RX90_pulse(
#             qubit, start=0, relative_phase=np.pi / 2
#         )

#         if flux_pulse_shapes and len(flux_pulse_shapes) == len(qubits):
#             flux_pulse_shape = eval(flux_pulse_shapes[qubit])
#         else:
#             flux_pulse_shape = Rectangular()

#         # wait before applying flux pulse (delay_before_flux)

#         # apply a detuning flux pulse
#         flux_pulses[qubit] = FluxPulse(
#             start=initial_RY90_pulses[qubit].se_finish + delay_before_flux,
#             duration=flux_pulse_duration_end,  # sweep to produce oscillations
#             amplitude=flux_pulse_amplitude,  # fix for each run
#             shape=flux_pulse_shape,
#             channel=platform.qubits[qubit].flux.name,
#             qubit=platform.qubits[qubit].name,
#         )

#         # wait after applying flux pulse (delay_after_flux)

#         # rotate around the X asis Rx(-pi/2) to meassure Y component
#         RX90_pulses[qubit] = platform.create_RX90_pulse(
#             qubit,
#             start=flux_pulses[qubit].finish + delay_after_flux,
#             relative_phase=np.pi,
#         )

#         # rotate around the Y asis Ry(pi/2) to meassure X component
#         RY90_pulses[qubit] = platform.create_RX90_pulse(
#             qubit,
#             start=flux_pulses[qubit].finish + delay_after_flux,
#             relative_phase=np.pi / 2,
#         )

#         # add ro pulse at the end of each sequence
#         MZ_ro_pulses[qubit] = platform.create_qubit_readout_pulse(
#             qubit, start=RX90_pulses[qubit].finish
#         )

#         # add pulses to the sequences
#         MX_seq.add(
#             initial_RY90_pulses[qubit],
#             flux_pulses[qubit],
#             RY90_pulses[qubit],
#             MZ_ro_pulses[qubit],
#         )
#         MY_seq.add(
#             initial_RY90_pulses[qubit],
#             flux_pulses[qubit],
#             RX90_pulses[qubit],
#             MZ_ro_pulses[qubit],
#         )
#         MZ_seq.add(
#             initial_RY90_pulses[qubit],
#             flux_pulses[qubit],
#             MZ_ro_pulses[qubit],
#         )
#         M0_seq.add(
#             MZ_ro_pulses[qubit],
#         )
#         M1_seq.add(
#             platform.create_RX_pulse(qubit, start=0, relative_phase=np.pi / 2),
#             MZ_ro_pulses[qubit],
#         )

#         # DEBUG: Plot Cryoscope Sequences
#         # MX_seq.plot("MX_seq")
#         # MY_seq.plot("MY_seq")
#         # MZ_seq.plot("MZ_seq")
#         # M0_seq.plot("M0_seq")
#         # M1_seq.plot("M1_seq")

#     MX_tag = "MX"
#     MY_tag = "MY"
#     MZ_tag = "MZ"
#     M0_tag = "M0"
#     M1_tag = "M1"

#     # define the parameters to sweep and their range:
#     # flux pulse amplitude
#     # flux pulse duration

#     flux_pulse_amplitude_range = np.arange(
#         flux_pulse_amplitude_start, flux_pulse_amplitude_end, flux_pulse_amplitude_step
#     )
#     amplitude_sweeper = Sweeper(
#         Parameter.amplitude,
#         flux_pulse_amplitude_range,
#         pulses=[flux_pulses[qubit] for qubit in qubits],
#     )
#     flux_pulse_duration_range = np.arange(
#         flux_pulse_duration_start, flux_pulse_duration_end, flux_pulse_duration_step
#     )
#     duration_sweeper = Sweeper(
#         Parameter.duration,
#         flux_pulse_duration_range,
#         pulses=[flux_pulses[qubit] for qubit in qubits],
#     )

#     # create a DataUnits object to store the results,
#     # DataUnits stores by default MSR, phase, i, q
#     # additionally include flux pulse duration, amplitude and the probability
#     data = DataUnits(
#         name=f"data",
#         quantities={
#             "flux_pulse_duration": "ns",
#             "flux_pulse_amplitude": "dimensionless",
#             "prob": "dimensionless",
#         },
#         options=["component", "qubit", "iteration"],
#     )

#     durations = np.repeat(flux_pulse_duration_range, len(flux_pulse_amplitude_range))
#     amplitudes = np.tile(flux_pulse_amplitude_range, len(flux_pulse_duration_range))

#     ndata = len(flux_pulse_amplitude_range) * len(flux_pulse_duration_range)
#     for iteration in range(software_averages):
#         # execute the pulse sequences
#         for sequence, tag in [
#             (MX_seq, MX_tag),
#             (MY_seq, MY_tag),
#             (MZ_seq, MZ_tag),
#             (M0_seq, M0_tag),
#             (M1_seq, M1_tag),
#         ]:
#             results = platform.sweep(
#                 sequence,
#                 duration_sweeper,
#                 amplitude_sweeper,
#                 nshots=nshots,
#                 relaxation_time=relaxation_time,
#                 average=True,
#             )
#             for qubit in qubits:
#                 qubit_res = results[MZ_ro_pulses[qubit].serial].raw
#                 qubit_res.update(
#                     {
#                         "prob[dimensionless]": results[
#                             MZ_ro_pulses[qubit].serial
#                         ].shots,
#                         "flux_pulse_duration[ns]": durations,
#                         "flux_pulse_amplitude[dimensionless]": amplitudes,
#                         "component": ndata * [tag],
#                         "qubit": ndata * [qubit],
#                         "iteration": ndata * [iteration],
#                     }
#                 )
#                 data.add_data_from_dict(qubit_res)

#     # finally, save the remaining data
#     yield data

#     import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# from lmfit import Model
# from plotly.subplots import make_subplots
# from scipy.optimize import curve_fit
# from scipy.signal import argrelextrema, lfilter, savgol_filter

# from qibocal.data import Data, DataUnits  # to be removed
# from qibocal.fitting.utils import cos, exp, flipping, lorenzian, rabi, ramsey
# from qibocal.plots.utils import get_color, get_color_pair, get_data_subfolders

# MX_tag = "MX"
# MY_tag = "MY"
# MZ_tag = "MZ"
# M0_tag = "M0"
# M1_tag = "M1"

# values_column = "prob"


# def load_data(folder, subfolder, routine, name):
#     file = f"{folder}/{subfolder}/{routine}/{name}.csv"
#     # skip loading units row
#     data: pd.DataFrame = pd.read_csv(file, header=[0], skiprows=[1], index_col=[0])

#     # drop unnecessary columns and average over iterations
#     data = (
#         data.drop(columns=["i", "q", "phase", "iteration"])
#         .groupby(
#             ["flux_pulse_duration", "flux_pulse_amplitude", "component", "qubit"],
#             as_index=False,
#         )
#         .mean()
#     )
#     return data


# def normalise_1(df: pd.DataFrame):
#     # normalise between -1 to 1 both X & Y symultaneously
#     # correct XY misalignment

#     df = df.copy()
#     for qubit in df["qubit"].unique():
#         qubit_m = df["qubit"] == qubit
#         durations, amplitudes = get_flux_pulse_durations_and_amplitudes(df[qubit_m])
#         MX_m = df["component"] == MX_tag
#         MY_m = df["component"] == MY_tag

#         for amp in amplitudes:
#             amp_m = df["flux_pulse_amplitude"] == amp

#             # normalise between -1 to 1
#             dfqa = df.loc[qubit_m & amp_m, values_column]
#             if (dfqa.size != 0) and (np.max(dfqa) != np.min(dfqa)):
#                 df.loc[qubit_m & amp_m, values_column] = (
#                     (dfqa - np.min(dfqa)) / (np.max(dfqa) - np.min(dfqa))
#                 ) * 2 - 1

#             # correct XY misalignment
#             delta = (
#                 np.mean(
#                     df.loc[qubit_m & amp_m & MY_m, values_column].to_numpy()
#                     - df.loc[qubit_m & amp_m & MX_m, values_column].to_numpy()
#                 )
#                 / 2
#             )
#             df.loc[qubit_m & amp_m & MX_m, values_column] += delta
#             df.loc[qubit_m & amp_m & MY_m, values_column] -= delta
#     return df


# def normalise_2(df: pd.DataFrame):
#     # normalise between -1 to 1 each X & Y separately
#     # correct XY misalignment

#     df = df.copy()
#     for qubit in df["qubit"].unique():
#         qubit_m = df["qubit"] == qubit
#         durations, amplitudes = get_flux_pulse_durations_and_amplitudes(df[qubit_m])
#         MX_m = df["component"] == MX_tag
#         MY_m = df["component"] == MY_tag

#         for amp in amplitudes:
#             amp_m = df["flux_pulse_amplitude"] == amp
#             # normalise between -1 to 1
#             for component_m in [MX_m, MY_m]:
#                 dfqa = df.loc[qubit_m & component_m & amp_m, values_column]
#                 if (dfqa.size != 0) and (np.max(dfqa) != np.min(dfqa)):
#                     df.loc[qubit_m & component_m & amp_m, values_column] = (
#                         (dfqa - np.min(dfqa)) / (np.max(dfqa) - np.min(dfqa))
#                     ) * 2 - 1

#             # correct XY misalignment
#             delta = (
#                 np.mean(
#                     df.loc[qubit_m & amp_m & MY_m, values_column].to_numpy()
#                     - df.loc[qubit_m & amp_m & MX_m, values_column].to_numpy()
#                 )
#                 / 2
#             )
#             df.loc[qubit_m & amp_m & MX_m, values_column] += delta
#             df.loc[qubit_m & amp_m & MY_m, values_column] -= delta
#     return df


# def normalise_3(df: pd.DataFrame):
#     # normalise between -1 to 1 both X & Y symultaneously
#     df = df.copy()
#     for qubit in df["qubit"].unique():
#         qubit_m = df["qubit"] == qubit
#         durations, amplitudes = get_flux_pulse_durations_and_amplitudes(df[qubit_m])

#         for amp in amplitudes:
#             amp_m = df["flux_pulse_amplitude"] == amp
#             # normalise between -1 to 1
#             dfqa = df.loc[qubit_m & amp_m, values_column]
#             if (dfqa.size != 0) and (np.max(dfqa) != np.min(dfqa)):
#                 df.loc[qubit_m & amp_m, values_column] = (
#                     (dfqa - np.min(dfqa)) / (np.max(dfqa) - np.min(dfqa))
#                 ) * 2 - 1
#     return df


# def normalise_4(df: pd.DataFrame):
#     # normalise between -1 to 1 each X & Y separately
#     df = df.copy()
#     for qubit in df["qubit"].unique():
#         qubit_m = df["qubit"] == qubit
#         durations, amplitudes = get_flux_pulse_durations_and_amplitudes(df[qubit_m])
#         MX_m = df["component"] == MX_tag
#         MY_m = df["component"] == MY_tag

#         for amp in amplitudes:
#             amp_m = df["flux_pulse_amplitude"] == amp
#             # normalise between -1 to 1
#             for component_m in [MX_m, MY_m]:
#                 dfqa = df.loc[qubit_m & component_m & amp_m, values_column]
#                 if (dfqa.size != 0) and (np.max(dfqa) != np.min(dfqa)):
#                     df.loc[qubit_m & component_m & amp_m, values_column] = (
#                         (dfqa - np.min(dfqa)) / (np.max(dfqa) - np.min(dfqa))
#                     ) * 2 - 1
#     return df


# def normalise(df: pd.DataFrame):
#     # correct XY misalignment
#     # normalise between -1 to 1 both X & Y symultaneously

#     df = df.copy()
#     for qubit in df["qubit"].unique():
#         qubit_m = df["qubit"] == qubit
#         durations, amplitudes = get_flux_pulse_durations_and_amplitudes(df[qubit_m])
#         MX_m = df["component"] == MX_tag
#         MY_m = df["component"] == MY_tag

#         for amp in amplitudes:
#             amp_m = df["flux_pulse_amplitude"] == amp

#             # correct XY misalignment
#             delta = (
#                 np.mean(
#                     df.loc[qubit_m & amp_m & MY_m, values_column].to_numpy()
#                     - df.loc[qubit_m & amp_m & MX_m, values_column].to_numpy()
#                 )
#                 / 2
#             )
#             df.loc[qubit_m & amp_m & MX_m, values_column] += delta
#             df.loc[qubit_m & amp_m & MY_m, values_column] -= delta

#             # normalise between -1 to 1
#             dfqa = df.loc[qubit_m & amp_m, values_column]
#             if (dfqa.size != 0) and (np.max(dfqa) != np.min(dfqa)):
#                 df.loc[qubit_m & amp_m, values_column] = (
#                     (dfqa - np.min(dfqa)) / (np.max(dfqa) - np.min(dfqa))
#                 ) * 2 - 1
#     return df


# def get_flux_pulse_durations_and_amplitudes(df: pd.DataFrame):
#     MX_m = df["component"] == MX_tag
#     MY_m = df["component"] == MY_tag
#     durations = np.intersect1d(
#         df["flux_pulse_duration"][MX_m], df["flux_pulse_duration"][MY_m]
#     )
#     amplitudes = np.intersect1d(
#         df["flux_pulse_amplitude"][MX_m], df["flux_pulse_amplitude"][MY_m]
#     )
#     return durations, amplitudes


# def get_values(df: pd.DataFrame, component=None, duration=None, amplitude=None):
#     values = df[values_column]
#     if component:
#         values = values[df["component"] == component]
#     if type(duration) == np.ndarray:
#         values = values[df["flux_pulse_duration"].isin(duration.tolist())]
#     elif duration:
#         values = values[df["flux_pulse_duration"] == duration]
#     if type(amplitude) == np.ndarray:
#         values = values[df["flux_pulse_amplitude"].isin(amplitude.tolist())]
#     elif amplitude:
#         values = values[df["flux_pulse_amplitude"] == amplitude]
#     return values.to_numpy()


# def flux_pulse_timing(folder, routine, qubit, format):
#     figs = {}

#     figs["flux_pulse_timing"] = make_subplots(
#         rows=1,
#         cols=1,
#         horizontal_spacing=0.1,
#         vertical_spacing=0.1,
#     )

#     # iterate over multiple data folders
#     subfolders = get_data_subfolders(folder)
#     report_n = 0
#     for subfolder in subfolders:
#         try:
#             data = DataUnits.load_data(folder, subfolder, routine, format, "data")
#             data.df = (
#                 data.df[data.df["qubit"] == qubit]
#                 .drop(columns=["qubit", "iteration", "i", "q", "phase"])
#                 .groupby(
#                     ["flux_pulse_amplitude", "flux_pulse_start"],
#                     as_index=False,
#                 )
#                 .mean()
#             )
#         except:
#             data = DataUnits(
#                 name=f"data",
#                 quantities={
#                     "flux_pulse_amplitude": "dimensionless",
#                     "flux_pulse_start": "ns",
#                     "prob": "dimensionless",
#                 },
#                 options=["qubit", "iteration"],
#             )

#         amplitudes = (
#             data.df["flux_pulse_amplitude"]
#             .pint.to("dimensionless")
#             .pint.magnitude.unique()
#         )
#         starts = data.df["flux_pulse_start"].pint.to("ns").pint.magnitude.unique()

#     for amp in amplitudes:
#         figs["flux_pulse_timing"].add_trace(
#             go.Scatter(
#                 x=starts,
#                 y=data.df[data.df["flux_pulse_amplitude"] == amp]["MSR"]
#                 .pint.to("V")
#                 .pint.magnitude.to_numpy(),
#                 name=f"q{qubit}: <X> | A = {amp:.3f}",
#             ),
#             row=1,
#             col=1,
#         )

#     figs["flux_pulse_timing"].update_layout(
#         xaxis_title="Flux start time (ns)",
#         yaxis_title="Magnitude X component (dimensionless)",
#     )
#     fitting_report = "No fitting data"
#     return [figs["flux_pulse_timing"]], fitting_report


# def cryoscope_raw(folder, routine, qubit, format):
#     show_not_corrected = False
#     show_calibration_points = False

#     # load qubit data
#     df = load_data(folder, "data", routine, "data")
#     df = df[df["qubit"] == qubit]
#     df_not_corrected = normalise_3(df)
#     df = normalise_1(df)
#     durations, amplitudes = get_flux_pulse_durations_and_amplitudes(df)

#     figs = {}
#     figs["raw"] = make_subplots(
#         rows=1,
#         cols=1,
#         horizontal_spacing=0.1,
#         vertical_spacing=0.1,
#     )

#     opacity = 0.3

#     for n, amp in enumerate(amplitudes):
#         figs["raw"].add_trace(
#             go.Scatter(
#                 x=durations,
#                 y=get_values(df, MX_tag, amplitude=amp),
#                 name=f"q{qubit}: <X> | A = {amp:.3f}",
#                 marker_color=get_color_pair(n)[0],
#             ),
#             row=1,
#             col=1,
#         )

#         figs["raw"].add_trace(
#             go.Scatter(
#                 x=durations,
#                 y=get_values(df, MY_tag, amplitude=amp),
#                 name=f"q{qubit}: <Y> | A = {amp:.3f}",
#                 marker_color=get_color_pair(n)[1],
#             ),
#             row=1,
#             col=1,
#         )

#         if show_not_corrected:
#             figs["raw"].add_trace(
#                 go.Scatter(
#                     x=durations,
#                     y=get_values(df_not_corrected, MX_tag, amplitude=amp),
#                     name=f"q{qubit}: <X> | A = {amp:.3f}",
#                     marker_color="Grey",
#                     opacity=opacity,
#                 ),
#                 row=1,
#                 col=1,
#             )

#             figs["raw"].add_trace(
#                 go.Scatter(
#                     x=durations,
#                     y=get_values(df_not_corrected, MY_tag, amplitude=amp),
#                     name=f"q{qubit}: <Y> | A = {amp:.3f}",
#                     marker_color="Grey",
#                     opacity=opacity,
#                 ),
#                 row=1,
#                 col=1,
#             )
#             window_length = len(durations) // 24 * 2 + 3
#             smoothed_average = savgol_filter(
#                 (
#                     get_values(df, MX_tag, amplitude=amp)
#                     + get_values(df, MY_tag, amplitude=amp)
#                 )
#                 / 2,
#                 window_length=window_length,
#                 polyorder=3,
#             )

#             figs["raw"].add_trace(
#                 go.Scatter(
#                     x=durations,
#                     y=smoothed_average,
#                     name=f"q{qubit}: <smoothed_average> | A = {amp:.3f}",
#                     marker_color=get_color(1),
#                     opacity=opacity,
#                 ),
#                 row=1,
#                 col=1,
#             )

#         # figs["raw"].add_trace(
#         #     go.Scatter(
#         #         x=durations,
#         #         y=get_prob(df, MY_tag, amplitude=amp) - get_prob(df, MX_tag, amplitude=amp),
#         #         name=f"q{qubit}: <difs> | A = {amp:.3f}",
#         #         marker_color=get_color(1),
#         #         opacity=opacity,
#         #     ),
#         #     row=1,
#         #     col=1,
#         # )
#         if show_calibration_points:
#             figs["raw"].add_trace(
#                 go.Scatter(
#                     x=durations,
#                     y=get_values(df, MZ_tag, amplitude=amp),
#                     name=f"q{qubit}: <Z> | A = {amp:.3f}",
#                     marker_color=get_color(0),
#                     opacity=opacity,
#                     legendgroup=f"q{qubit} | A = {amp:.3f}",
#                 ),
#                 row=1,
#                 col=1,
#             )

#             figs["raw"].add_trace(
#                 go.Scatter(
#                     x=durations,
#                     y=get_values(df, M0_tag, amplitude=amp),
#                     marker_color=get_color(0),
#                     opacity=opacity,
#                     showlegend=False,
#                     legendgroup=f"q{qubit} | A = {amp:.3f}",
#                 ),
#                 row=1,
#                 col=1,
#             )

#             figs["raw"].add_trace(
#                 go.Scatter(
#                     x=durations,
#                     y=get_values(df, M1_tag, amplitude=amp),
#                     marker_color=get_color(0),
#                     opacity=opacity,
#                     showlegend=False,
#                     legendgroup=f"q{qubit} | A = {amp:.3f}",
#                 ),
#                 row=1,
#                 col=1,
#             )

#     figs["raw"].update_layout(
#         xaxis_title="Pulse duration (ns)",
#         yaxis_title="Magnitude (dimensionless)",
#         title=f"Raw data",
#     )
#     fitting_report = "No fitting data"
#     return [figs["raw"]], fitting_report


# def cryoscope_dephasing(folder, routine, qubit, format):
#     # load qubit data
#     df = load_data(folder, "data", routine, "data")
#     df = df[df["qubit"] == qubit]
#     df = normalise(df)
#     durations, amplitudes = get_flux_pulse_durations_and_amplitudes(df)

#     figs = {}
#     figs["norm"] = make_subplots(
#         rows=1,
#         cols=1,
#         horizontal_spacing=0.1,
#         vertical_spacing=0.2,
#     )

#     for amp in amplitudes:
#         re = get_values(df, MX_tag, amplitude=amp)
#         im = get_values(df, MY_tag, amplitude=amp)
#         magnitude = np.abs(re + 1j * im)

#         figs["norm"].add_trace(
#             go.Scatter(
#                 x=durations,
#                 y=magnitude,
#                 name=f"q{qubit}: <magnitude> | A = {amp:.3f}",
#             ),
#             row=1,
#             col=1,
#         )

#         window_length = len(durations) // 24 * 2 + 3
#         smoothed_average = savgol_filter(
#             magnitude,
#             window_length=window_length,
#             polyorder=3,
#         )
#         figs["norm"].add_trace(
#             go.Scatter(
#                 x=durations,
#                 y=smoothed_average,
#                 name=f"q{qubit}: <smooth> | A = {amp:.3f}",
#             ),
#             row=1,
#             col=1,
#         )

#         dephased_point = np.argmax(
#             savgol_filter(magnitude, window_length=window_length, polyorder=3, deriv=2)
#         )
#         figs["norm"].add_vline(
#             x=dephased_point, line_width=1, line_dash="dash", line_color="red"
#         )

#     fitting_report = "No fitting data"
#     return [figs["norm"]], fitting_report


# def cryoscope_dephasing_heatmap(folder, routine, qubit, format):
#     # load qubit data
#     df = load_data(folder, "data", routine, "data")
#     df = df[df["qubit"] == qubit]
#     df = normalise(df)
#     durations, amplitudes = get_flux_pulse_durations_and_amplitudes(df)

#     figs = {}
#     figs["norm"] = make_subplots(
#         rows=1,
#         cols=1,
#         horizontal_spacing=0.1,
#         vertical_spacing=0.2,
#     )

#     re = get_values(df, MX_tag, amplitude=amplitudes)
#     im = get_values(df, MY_tag, amplitude=amplitudes)

#     signal = re + 1j * im
#     signal = signal.reshape((len(durations), len(amplitudes))).transpose()

#     window_length = len(durations) // 24 * 2 + 3

#     dephased_points = np.array([])
#     for row in signal:
#         dephased_points = np.append(
#             dephased_points,
#             np.argmax(
#                 savgol_filter(
#                     np.abs(row), window_length=window_length, polyorder=3, deriv=2
#                 )
#             ),
#         )

#     global_title = "Dephasing"
#     title_x = "Flux Pulse Duration"
#     title_y = "Amplitudes"
#     figs["norm"].add_trace(
#         go.Heatmap(
#             x=durations,
#             y=amplitudes,
#             z=np.abs(signal),
#             colorbar=dict(len=0.46, y=0.25),
#         ),
#         row=1,
#         col=1,
#     )

#     figs["norm"].add_trace(
#         go.Scatter(
#             x=dephased_points,
#             y=amplitudes,
#             mode="markers+lines",
#             marker=dict(size=3, color="red"),
#         ),
#         row=1,
#         col=1,
#     )

#     figs["norm"].update_layout(
#         xaxis_title=title_x,
#         yaxis_title=title_y,
#         title=f"{global_title}",
#     )
#     fitting_report = "No fitting data"
#     return [figs["norm"]], fitting_report


# def cryoscope_fft_peak_fitting(folder, routine, qubit, format):
#     # load qubit data
#     df = load_data(folder, "data", routine, "data")
#     df = df[df["qubit"] == qubit]
#     df = normalise(df)
#     durations, amplitudes = get_flux_pulse_durations_and_amplitudes(df)

#     figs = {}
#     figs["norm"] = make_subplots(
#         rows=1,
#         cols=1,
#         horizontal_spacing=0.1,
#         vertical_spacing=0.2,
#     )

#     re = get_values(df, MX_tag, amplitude=amplitudes)
#     im = get_values(df, MY_tag, amplitude=amplitudes)

#     signal = re + 1j * im
#     signal = signal.reshape((len(durations), len(amplitudes))).transpose()

#     window_length = len(durations) // 24 * 2 + 3
#     # calculate the duration where the state is dephased
#     dephased_points = np.array([])
#     for row in signal:
#         dephased_points = np.append(
#             dephased_points,
#             np.argmax(
#                 savgol_filter(
#                     np.abs(row), window_length=window_length, polyorder=3, deriv=2
#                 )
#             ),
#         )

#     # # trim data beyond that point
#     # cut_off_duration = np.argmax(durations>=np.mean(dephased_points))
#     # durations = durations[:cut_off_duration]
#     # signal = signal[:,:cut_off_duration]

#     window_length = len(durations) // 24 * 2 + 3

#     # fft
#     fft = np.zeros(signal.shape)

#     curve_amps = np.array([])
#     curve_detunings = np.array([])

#     num_points = len(durations)
#     interval = (durations[1] - durations[0]) * 1e-9
#     amp_freqs = np.fft.fftfreq(n=num_points, d=interval)
#     amp_freqs[amp_freqs >= 10e6] -= 1 / interval
#     mask = np.argsort(amp_freqs)
#     amp_freqs = amp_freqs[mask]
#     freqs = np.tile(amp_freqs, (len(amplitudes), 1))

#     for row_n, amp in enumerate(amplitudes):
#         amp_signal = signal[row_n, :]
#         amp_fft = np.fft.fft(amp_signal - np.mean(amp_signal))[mask]
#         fft[row_n] = amp_fft

#         figs["norm"].add_trace(
#             go.Scatter(x=amp_freqs, y=np.abs(amp_fft), name=f"fft | A = {amp:.3f}"),
#             row=1,
#             col=1,
#         )

#         smoothed_y = smooth(np.abs(amp_fft), window_len=window_length)
#         percval = np.percentile(
#             smoothed_y, 90
#         )  # finds the top perc points in amplitude smoothed_y
#         filtered_y = np.where(smoothed_y > percval, smoothed_y, percval)
#         array_peaks = argrelextrema(filtered_y, np.greater)
#         peaks_x = np.arange(num_points)[array_peaks]
#         sort_mask = np.argsort(np.absolute(amp_fft)[array_peaks])[::-1]

#         if len(peaks_x[sort_mask]) > 0 and amp_freqs[peaks_x[sort_mask][0]] <= -30e6:
#             curve_amps = np.append(curve_amps, amp)
#             curve_detunings = np.append(
#                 curve_detunings, amp_freqs[peaks_x[sort_mask][0]]
#             )

#         figs["norm"].add_trace(
#             go.Scatter(x=amp_freqs, y=smoothed_y, name=f"smoothed_y | A = {amp:.3f}"),
#             row=1,
#             col=1,
#         )

#         figs["norm"].add_trace(
#             go.Scatter(
#                 x=amp_freqs,
#                 y=filtered_y,
#                 name=f"filtered_y | A = {amp:.3f} | percentile = {percval:.3f}",
#             ),
#             row=1,
#             col=1,
#         )

#     global_title = "FFT Peak Fitting"
#     title_x = "Detunning (GHz)"
#     title_y = "Spectral Density"

#     figs["norm"].update_layout(
#         xaxis_title=title_x,
#         yaxis_title=title_y,
#         title=f"{global_title}",
#     )
#     fitting_report = "No fitting data"
#     return [figs["norm"]], fitting_report


# def cryoscope_fft(folder, routine, qubit, format):
#     # load qubit data
#     df = load_data(folder, "data", routine, "data")
#     df = df[df["qubit"] == qubit]
#     df = normalise(df)
#     durations, amplitudes = get_flux_pulse_durations_and_amplitudes(df)

#     figs = {}
#     figs["norm"] = make_subplots(
#         rows=1,
#         cols=1,
#         horizontal_spacing=0.1,
#         vertical_spacing=0.2,
#     )

#     re = get_values(df, MX_tag, amplitude=amplitudes)
#     im = get_values(df, MY_tag, amplitude=amplitudes)

#     signal = re + 1j * im
#     signal = signal.reshape((len(durations), len(amplitudes))).transpose()

#     window_length = len(durations) // 24 * 2 + 3
#     # calculate the duration where the state is dephased
#     dephased_points = np.array([])
#     for row in signal:
#         dephased_points = np.append(
#             dephased_points,
#             np.argmax(
#                 savgol_filter(
#                     np.abs(row), window_length=window_length, polyorder=3, deriv=2
#                 )
#             ),
#         )

#     # # trim data beyond that point
#     # cut_off_duration = np.argmax(durations>=np.mean(dephased_points))
#     # durations = durations[:cut_off_duration]
#     # signal = signal[:,:cut_off_duration]

#     window_length = len(durations) // 24 * 2 + 3

#     # fft
#     fft = np.zeros(signal.shape)

#     curve_amps = np.array([])
#     curve_detunings = np.array([])

#     num_points = len(durations)
#     interval = (durations[1] - durations[0]) * 1e-9
#     amp_freqs = np.fft.fftfreq(n=num_points, d=interval)
#     amp_freqs[amp_freqs >= 30e6] -= 1 / interval  # FIXME
#     mask = np.argsort(amp_freqs)
#     amp_freqs = amp_freqs[mask]
#     freqs = np.tile(amp_freqs, (len(amplitudes), 1))

#     for row_n, amp in enumerate(amplitudes):
#         amp_signal = signal[row_n, :]
#         amp_fft = np.fft.fft(amp_signal - np.mean(amp_signal))[mask]

#         smoothed_y = smooth(np.abs(amp_fft), window_len=window_length)
#         percval = np.percentile(
#             smoothed_y, 90
#         )  # finds the top perc points in amplitude smoothed_y
#         filtered_y = np.where(smoothed_y > percval, smoothed_y, percval)
#         fft[row_n] = amp_fft
#         # fft[row_n] = smoothed_y

#         array_peaks = argrelextrema(filtered_y, np.greater)
#         peaks_x = np.arange(num_points)[array_peaks]
#         sort_mask = np.argsort(np.absolute(amp_fft)[array_peaks])[::-1]

#         if len(peaks_x[sort_mask]) > 0 and amp_freqs[peaks_x[sort_mask][0]] <= -30e6:
#             curve_amps = np.append(curve_amps, amp)
#             curve_detunings = np.append(
#                 curve_detunings, amp_freqs[peaks_x[sort_mask][0]]
#             )

#     pfit = np.polyfit(curve_amps, curve_detunings, 2)

#     print(f"f = {pfit[0]}*amp**2 + {pfit[1]}*amp + {pfit[2]}")

#     global_title = "FFT with curve fit"
#     title_x = "Detunning (GHz)"
#     title_y = "Amplitudes"
#     figs["norm"].add_trace(
#         go.Heatmap(
#             x=amp_freqs,
#             y=amplitudes,
#             z=np.abs(fft),
#         ),
#         row=1,
#         col=1,
#     )
#     figs["norm"].add_trace(
#         go.Scatter(
#             x=curve_detunings,
#             y=curve_amps,
#             name=f"Data points",
#             mode="markers",
#             marker=dict(size=5),
#         ),
#         row=1,
#         col=1,
#     )
#     figs["norm"].add_trace(
#         go.Scatter(
#             x=pfit[0] * curve_amps**2 + pfit[1] * curve_amps + pfit[2],
#             y=curve_amps,
#             name=f"Curve fit",
#         ),
#         row=1,
#         col=1,
#     )
#     figs["norm"].update_layout(
#         xaxis_title=title_x,
#         yaxis_title=title_y,
#         title=f"{global_title}",
#     )
#     fitting_report = "No fitting data"
#     return [figs["norm"]], fitting_report


# def cryoscope_phase(folder, routine, qubit, format):
#     # load qubit data
#     df = load_data(folder, "data", routine, "data")
#     df = df[df["qubit"] == qubit]
#     df = normalise(df)
#     durations, amplitudes = get_flux_pulse_durations_and_amplitudes(df)

#     figs = {}
#     figs["norm"] = make_subplots(
#         rows=1,
#         cols=1,
#         horizontal_spacing=0.1,
#         vertical_spacing=0.2,
#     )

#     for amp in amplitudes:
#         re = get_values(df, MX_tag, amplitude=amp)
#         im = get_values(df, MY_tag, amplitude=amp)

#         phase = np.arctan2(im, re)

#         # Phase vs. Time
#         global_title = "Phase vs. Time"
#         title_x = "Flux Pulse duration"
#         title_y = "Phase"
#         figs["norm"].add_trace(
#             go.Scatter(
#                 x=durations,
#                 y=phase,
#                 name=f"Phase | A = {amp:.3f}",
#                 mode="lines+markers",
#                 marker=dict(size=5),
#             ),
#             row=1,
#             col=1,
#         )

#     figs["norm"].update_layout(
#         xaxis_title=title_x,
#         yaxis_title=title_y,
#         title=f"{global_title}",
#     )
#     fitting_report = "No fitting data"
#     return [figs["norm"]], fitting_report


# def cryoscope_phase_heatmap(folder, routine, qubit, format):
#     # load qubit data
#     df = load_data(folder, "data", routine, "data")
#     df = df[df["qubit"] == qubit]
#     df = normalise(df)
#     durations, amplitudes = get_flux_pulse_durations_and_amplitudes(df)

#     figs = {}
#     figs["norm"] = make_subplots(
#         rows=1,
#         cols=1,
#         horizontal_spacing=0.1,
#         vertical_spacing=0.2,
#     )

#     re = get_values(df, MX_tag, amplitude=amplitudes)
#     im = get_values(df, MY_tag, amplitude=amplitudes)

#     signal = re + 1j * im
#     signal = signal.reshape((len(durations), len(amplitudes))).transpose()
#     phase = np.arctan2(signal.imag, signal.real)

#     global_title = "Phase vs. Time"
#     title_x = "Flux Pulse duration"
#     title_y = "Amplitude"
#     figs["norm"].add_trace(
#         go.Heatmap(
#             x=durations,
#             y=amplitudes,
#             z=phase,
#         ),
#         row=1,
#         col=1,
#     )

#     figs["norm"].update_layout(
#         xaxis_title=title_x,
#         yaxis_title=title_y,
#         title=f"{global_title}",
#     )
#     fitting_report = "No fitting data"
#     return [figs["norm"]], fitting_report


# def cryoscope_phase_unwrapped(folder, routine, qubit, format):
#     # load qubit data
#     df = load_data(folder, "data", routine, "data")
#     df = df[df["qubit"] == qubit]
#     df = normalise(df)
#     durations, amplitudes = get_flux_pulse_durations_and_amplitudes(df)

#     figs = {}
#     figs["norm"] = make_subplots(
#         rows=1,
#         cols=1,
#         horizontal_spacing=0.1,
#         vertical_spacing=0.2,
#     )

#     sampling_rate = 1 / (durations[1] - durations[0])

#     for amp in amplitudes:
#         re = get_values(df, MX_tag, amplitude=amp)
#         im = get_values(df, MY_tag, amplitude=amp)

#         magnitude = np.abs(re + 1j * im)
#         window_length = len(durations) // 24 * 2 + 3
#         dephased_point = np.argmax(
#             savgol_filter(magnitude, window_length=window_length, polyorder=3, deriv=2)
#         )
#         cut_off_duration = np.argmax(durations >= dephased_point)
#         re = re[:cut_off_duration]
#         im = im[:cut_off_duration]

#         phase = np.arctan2(im, re)
#         phase_unwrapped = np.unwrap(phase)
#         # convert possitive detunings into negative
#         if np.mean(np.diff(phase_unwrapped)) / (2 * np.pi) * sampling_rate * 1e9 > 0:
#             phase_unwrapped += -2 * np.pi * durations[:cut_off_duration]

#         # Phase vs. Time
#         global_title = "Phase Unwrapped vs. Time"
#         title_x = "Flux Pulse duration"
#         title_y = "Phase"
#         figs["norm"].add_trace(
#             go.Scatter(
#                 x=durations[:cut_off_duration],
#                 y=phase_unwrapped,
#                 name=f"Phase | A = {amp:.3f}",
#             ),
#             row=1,
#             col=1,
#         )

#     figs["norm"].update_layout(
#         xaxis_title=title_x,
#         yaxis_title=title_y,
#         title=f"{global_title}",
#     )
#     fitting_report = "No fitting data"
#     return [figs["norm"]], fitting_report


# def cryoscope_phase_unwrapped_heatmap(folder, routine, qubit, format):
#     # load qubit data
#     df = load_data(folder, "data", routine, "data")
#     df = df[df["qubit"] == qubit]
#     df = normalise(df)
#     durations, amplitudes = get_flux_pulse_durations_and_amplitudes(df)

#     figs = {}
#     figs["norm"] = make_subplots(
#         rows=1,
#         cols=1,
#         horizontal_spacing=0.1,
#         vertical_spacing=0.2,
#     )

#     sampling_rate = 1 / (durations[1] - durations[0])

#     re = get_values(df, MX_tag, amplitude=amplitudes)
#     im = get_values(df, MY_tag, amplitude=amplitudes)

#     signal = re + 1j * im
#     signal = signal.reshape((len(durations), len(amplitudes))).transpose()

#     phase = np.arctan2(signal.imag, signal.real)
#     phase_unwrapped = np.unwrap(phase, axis=1)

#     window_length = len(durations) // 24 * 2 + 3

#     dephased_points = np.array([])
#     for row in signal:
#         dephased_points = np.append(
#             dephased_points,
#             np.argmax(
#                 savgol_filter(
#                     np.abs(row), window_length=window_length, polyorder=3, deriv=2
#                 )
#             ),
#         )
#     # trim data beyond that point
#     cut_off_duration = np.argmax(durations >= np.mean(dephased_points))

#     for row_n in range(signal.shape[0]):
#         if (
#             np.mean(np.diff(phase_unwrapped[row_n, :cut_off_duration]))
#             / (2 * np.pi)
#             * sampling_rate
#             * 1e9
#             > 0
#         ):
#             phase_unwrapped[row_n, :cut_off_duration] += (
#                 -2 * np.pi * durations[:cut_off_duration]
#             )
#             phase_unwrapped[row_n, cut_off_duration:] += (
#                 -2 * np.pi * durations[cut_off_duration]
#             )

#     global_title = "Phase unwrapped along duration vs. Time"
#     title_x = "Flux Pulse duration"
#     title_y = "Phase"
#     figs["norm"].add_trace(
#         go.Heatmap(
#             x=durations,
#             y=amplitudes,
#             z=phase_unwrapped,
#         ),
#         row=1,
#         col=1,
#     )

#     figs["norm"].add_trace(
#         go.Scatter(
#             x=dephased_points,
#             y=amplitudes,
#             mode="markers+lines",
#             marker=dict(size=3, color="red"),
#         ),
#         row=1,
#         col=1,
#     )

#     figs["norm"].update_layout(
#         xaxis_title=title_x,
#         yaxis_title=title_y,
#         title=f"{global_title}",
#     )
#     fitting_report = "No fitting data"
#     return [figs["norm"]], fitting_report


# def cryoscope_phase_amplitude_unwrapped_heatmap(folder, routine, qubit, format):
#     # in order for this plot to work, the amplitude unwrapping needs to start from amplitude 0
#     # load qubit data
#     df = load_data(folder, "data", routine, "data")
#     df = df[df["qubit"] == qubit]
#     df = normalise(df)
#     durations, amplitudes = get_flux_pulse_durations_and_amplitudes(df)

#     # start the unroll desde amp = 0 (no phase shift)
#     # unroll in both directions

#     figs = {}
#     figs["norm"] = make_subplots(
#         rows=1,
#         cols=1,
#         horizontal_spacing=0.1,
#         vertical_spacing=0.2,
#     )

#     re = get_values(df, MX_tag, amplitude=amplitudes)
#     im = get_values(df, MY_tag, amplitude=amplitudes)

#     signal = re + 1j * im
#     signal = signal.reshape((len(durations), len(amplitudes))).transpose()

#     phase = np.arctan2(signal.imag, signal.real)
#     phase_unwrapped = np.unwrap(phase, axis=1)

#     global_title = "Phase unwrapped along amplitude vs. Time"
#     title_x = "Flux Pulse duration"
#     title_y = "Phase"
#     phase = np.arctan2(signal.imag, signal.real)
#     phase_unwrapped = np.unwrap(phase, axis=0)
#     figs["norm"].add_trace(
#         go.Heatmap(
#             x=durations,
#             y=amplitudes,
#             z=phase_unwrapped,
#         ),
#         row=1,
#         col=1,
#     )

#     figs["norm"].update_layout(
#         xaxis_title=title_x,
#         yaxis_title=title_y,
#         title=f"{global_title}",
#     )
#     fitting_report = "No fitting data"
#     return [figs["norm"]], fitting_report


# def cryoscope_detuning_time(folder, routine, qubit, format):
#     # load qubit data
#     df = load_data(folder, "data", routine, "data")
#     df = df[df["qubit"] == qubit]
#     df = normalise(df)
#     durations, amplitudes = get_flux_pulse_durations_and_amplitudes(df)

#     figs = {}
#     figs["norm"] = make_subplots(
#         rows=1,
#         cols=1,
#         horizontal_spacing=0.1,
#         vertical_spacing=0.2,
#     )

#     sampling_rate = 1 / (durations[1] - durations[0])

#     for amp in amplitudes:
#         re = get_values(df, MX_tag, amplitude=amp)
#         im = get_values(df, MY_tag, amplitude=amp)

#         magnitude = np.abs(re + 1j * im)
#         window_length = len(durations) // 24 * 2 + 3
#         dephased_point = np.argmax(
#             savgol_filter(magnitude, window_length=window_length, polyorder=3, deriv=2)
#         )
#         cut_off_duration = np.argmax(durations >= dephased_point)
#         re = re[:cut_off_duration]
#         im = im[:cut_off_duration]

#         phase = np.arctan2(im, re)
#         phase_unwrapped = np.unwrap(phase)

#         detuning = np.diff(phase_unwrapped) / (2 * np.pi) * sampling_rate * 1e9
#         # convert possitive detunings into negative
#         if np.mean(detuning):
#             phase_unwrapped += -2 * np.pi * durations[:cut_off_duration]
#             detuning = np.diff(phase_unwrapped) / (2 * np.pi) * sampling_rate * 1e9

#         # window_length = cut_off_duration // 24 * 2 + 3
#         # # Derivate
#         # detuning = (
#         #     savgol_filter(
#         #         phase_unwrapped / (2 * np.pi),
#         #         window_length=window_length,
#         #         polyorder=2,
#         #         deriv=1,
#         #     )
#         #     * sampling_rate
#         #     * 1e9
#         # )

#         # Detunning vs. Time
#         global_title = "Detuning vs. Time"
#         title_x = "Flux Pulse duration"
#         title_y = "Detunning"
#         figs["norm"].add_trace(
#             go.Scatter(
#                 x=durations[:cut_off_duration], y=detuning, name=f"A = {amp:.3f}"
#             ),
#             row=1,
#             col=1,
#         )

#     figs["norm"].update_layout(
#         xaxis_title=title_x,
#         yaxis_title=title_y,
#         title=f"{global_title}",
#     )
#     fitting_report = "No fitting data"
#     return [figs["norm"]], fitting_report


# def cryoscope_distorted_amplitude_time(folder, routine, qubit, format):
#     # load qubit data
#     df = load_data(folder, "data", routine, "data")
#     df = df[df["qubit"] == qubit]
#     df = normalise(df)
#     durations, amplitudes = get_flux_pulse_durations_and_amplitudes(df)

#     figs = {}
#     figs["norm"] = make_subplots(
#         rows=1,
#         cols=1,
#         horizontal_spacing=0.1,
#         vertical_spacing=0.2,
#     )

#     re = get_values(df, MX_tag, amplitude=amplitudes)
#     im = get_values(df, MY_tag, amplitude=amplitudes)

#     signal = re + 1j * im
#     signal = signal.reshape((len(durations), len(amplitudes))).transpose()

#     window_length = len(durations) // 24 * 2 + 3
#     # calculate the duration where the state is dephased
#     dephased_points = np.array([])
#     for row in signal:
#         dephased_points = np.append(
#             dephased_points,
#             np.argmax(
#                 savgol_filter(
#                     np.abs(row), window_length=window_length, polyorder=3, deriv=2
#                 )
#             ),
#         )

#     # # trim data beyond that point
#     # cut_off_duration = np.argmax(durations>=np.mean(dephased_points))
#     # durations = durations[:cut_off_duration]
#     # signal = signal[:,:cut_off_duration]

#     window_length = len(durations) // 24 * 2 + 3

#     # fft
#     fft = np.zeros(signal.shape)

#     curve_amps = np.array([])
#     curve_detunings = np.array([])

#     num_points = len(durations)
#     interval = (durations[1] - durations[0]) * 1e-9
#     amp_freqs = np.fft.fftfreq(n=num_points, d=interval)
#     amp_freqs[amp_freqs >= 30e6] -= 1 / interval
#     mask = np.argsort(amp_freqs)
#     amp_freqs = amp_freqs[mask]
#     freqs = np.tile(amp_freqs, (len(amplitudes), 1))

#     for row_n, amp in enumerate(amplitudes):
#         amp_signal = signal[row_n, :]
#         amp_fft = np.fft.fft(amp_signal - np.mean(amp_signal))[mask]

#         smoothed_y = smooth(np.abs(amp_fft), window_len=window_length)
#         percval = np.percentile(
#             smoothed_y, 90
#         )  # finds the top perc points in amplitude smoothed_y
#         filtered_y = np.where(smoothed_y > percval, smoothed_y, percval)
#         fft[row_n] = amp_fft
#         # fft[row_n] = smoothed_y

#         array_peaks = argrelextrema(filtered_y, np.greater)
#         peaks_x = np.arange(num_points)[array_peaks]
#         sort_mask = np.argsort(np.absolute(amp_fft)[array_peaks])[::-1]

#         if len(peaks_x[sort_mask]) > 0 and amp_freqs[peaks_x[sort_mask][0]] <= -30e6:
#             curve_amps = np.append(curve_amps, amp)
#             curve_detunings = np.append(
#                 curve_detunings, amp_freqs[peaks_x[sort_mask][0]]
#             )

#     pfit = np.polyfit(curve_amps, curve_detunings, 2)

#     sampling_rate = 1 / (durations[1] - durations[0])

#     for amp in amplitudes:
#         re = get_values(df, MX_tag, amplitude=amp)
#         im = get_values(df, MY_tag, amplitude=amp)

#         magnitude = np.abs(re + 1j * im)
#         window_length = len(durations) // 24 * 2 + 3
#         dephased_point = np.argmax(
#             savgol_filter(magnitude, window_length=window_length, polyorder=3, deriv=2)
#         )
#         cut_off_duration = np.argmax(durations >= dephased_point)
#         re = re[:cut_off_duration]
#         im = im[:cut_off_duration]

#         phase = np.arctan2(im, re)
#         phase_unwrapped = np.unwrap(phase)

#         detuning = np.diff(phase_unwrapped) / (2 * np.pi) * sampling_rate * 1e9
#         # convert possitive detunings into negative
#         if np.mean(detuning):
#             phase_unwrapped += -2 * np.pi * durations[:cut_off_duration]
#             detuning = np.diff(phase_unwrapped) / (2 * np.pi) * sampling_rate * 1e9

#         clipped_detuning = np.clip(
#             detuning,
#             a_max=-1 / 4 * (pfit[1] ** 2) / pfit[0] + pfit[2],
#             a_min=np.min(detuning),
#         )
#         plt.plot(detuning)
#         plt.plot(clipped_detuning)
#         plt.savefig("/nfs/users/alvaro.orgaz/clipped.png")
#         plt.close()

#         distorted_pulse = (
#             -pfit[1]
#             - np.sqrt(pfit[1] ** 2 - (4 * pfit[0] * (pfit[2] - clipped_detuning)))
#         ) / (2 * pfit[0])
#         # distorted_pulse = (-pfit[1] + np.sqrt((pfit[1]**2 - (4* pfit[0] * (pfit[2] - clipped_detuning)))))/(2*pfit[0])

#         global_title = "Distorted Pulse Amplitude vs. Time"
#         title_x = "Flux Pulse duration"
#         title_y = "Amplitude"
#         figs["norm"].add_trace(
#             go.Scatter(
#                 x=durations,
#                 y=distorted_pulse,
#                 name=f"A = {amp:.3f}",
#                 mode="lines+markers",
#                 marker=dict(size=5),
#             ),
#             row=1,
#             col=1,
#         )

#     figs["norm"].update_layout(
#         xaxis_title=title_x,
#         yaxis_title=title_y,
#         title=f"{global_title}",
#     )
#     fitting_report = "No fitting data"
#     return [figs["norm"]], fitting_report


# def cryoscope_reconstructed_amplitude_time(folder, routine, qubit, format):
#     # load qubit data
#     df = load_data(folder, "data", routine, "data")
#     df = df[df["qubit"] == qubit]
#     df = normalise(df)
#     durations, amplitudes = get_flux_pulse_durations_and_amplitudes(df)

#     figs = {}
#     figs["norm"] = make_subplots(
#         rows=1,
#         cols=1,
#         horizontal_spacing=0.1,
#         vertical_spacing=0.2,
#     )

#     signal = []  # TODO: replace lists with np arrays
#     fft = []
#     freqs = []

#     curve_amps = []
#     curve_detunings = []

#     num_points = durations.shape[0]
#     interval = (durations[1] - durations[0]) * 1e-9
#     amp_freqs = np.fft.fftfreq(n=num_points, d=interval)
#     mask = np.argsort(amp_freqs)
#     amp_freqs = amp_freqs[mask]

#     sampling_rate = 1 / (durations[1] - durations[0])
#     derivative_window_length = 7 / sampling_rate
#     derivative_window_size = max(3, int(derivative_window_length * sampling_rate))
#     derivative_window_size += (derivative_window_size + 1) % 2
#     derivative_window_size = 15
#     derivative_order = 3
#     nyquist_order = 0

#     for amp in amplitudes:
#         re = get_values(df, MX_tag, amplitude=amp)
#         im = get_values(df, MY_tag, amplitude=amp)

#         amp_signal = re + 1j * im
#         signal.append(amp_signal)

#         freqs.append(amp_freqs)

#         amp_fft = np.fft.fft(amp_signal - np.mean(amp_signal))[mask]
#         fft.append(amp_fft)

#         smoothed_y = smooth(np.absolute(amp_fft), window_len=21)
#         percval = np.percentile(
#             smoothed_y, 90
#         )  # finds the top perc points in amplitude smoothed_y
#         filtered_y = np.where(smoothed_y > percval, smoothed_y, percval)
#         array_peaks = argrelextrema(filtered_y, np.greater)
#         peaks_x = np.arange(num_points)[array_peaks]
#         sort_mask = np.argsort(np.absolute(amp_fft)[array_peaks])[::-1]

#         if (
#             len(peaks_x[sort_mask]) > 0
#         ):  # and amp_freqs[peaks_x[sort_mask][0]] <= -30e6:
#             curve_amps.append(amp)
#             curve_detunings.append(amp_freqs[peaks_x[sort_mask][0]])

#     pfit = np.polyfit(curve_amps, curve_detunings, 2)
#     curve_amps = np.array(curve_amps)

#     # x=pfit[0]*curve_amps**2 + pfit[1]*curve_amps + pfit[2]

#     for amp in amplitudes:
#         re = get_values(df, MX_tag, amplitude=amp)
#         im = get_values(df, MY_tag, amplitude=amp)

#         # Unwrap phase
#         phase = np.arctan2(im, re)
#         phase_unwrapped = np.unwrap(phase)

#         # -----------------      Unwrap along durations     ---------------------------
#         # phase_unwrapped_matrix = np.array(amplitudes).reshape(len(amplitudes),1)
#         # for dur in durations: # unwrap along amplitudes
#         #     re = get_prob(df, MX_tag, duration=dur)
#         #     im = get_prob(df, MY_tag, duration=dur)

#         #     # Unwrap phase
#         #     phase = np.arctan2(im, re)
#         #     phase_unwrapped = np.unwrap(phase)

#         #     phase_unwrapped_matrix = np.append(phase_unwrapped_matrix, phase_unwrapped, axis=1)

#         # for n in range(1, len(amplitudes)):
#         #     phase_unwrapped = phase_unwrapped_matrix[n]
#         # -----------------------------------------------------------------------------

#         # Derivate
#         detuning = (
#             savgol_filter(
#                 phase_unwrapped / (2 * np.pi),
#                 window_length=derivative_window_size,
#                 polyorder=derivative_order,
#                 deriv=1,
#             )
#             * sampling_rate
#             * 1e9
#         )  # TODO: fix units e28!!??
#         print(f"pfit: {pfit}")

#         clipped_detuning = np.clip(
#             detuning,
#             a_max=-1 / 4 * (pfit[1] ** 2) / pfit[0] + pfit[2],
#             a_min=np.min(detuning),
#         )
#         plt.plot(detuning)
#         plt.plot(clipped_detuning)
#         plt.savefig("/nfs/users/alvaro.orgaz/clipped.png")
#         plt.close()

#         distorted_pulse = (
#             -pfit[1]
#             - np.sqrt(pfit[1] ** 2 - (4 * pfit[0] * (pfit[2] - clipped_detuning)))
#         ) / (2 * pfit[0])
#         # distorted_pulse = (-pfit[1] + np.sqrt((pfit[1]**2 - (4* pfit[0] * (pfit[2] - clipped_detuning)))))/(2*pfit[0])

#         global_title = "Reconstructed Pulse Amplitude vs. Time"
#         title_x = "Flux Pulse duration"
#         title_y = "Amplitude"
#         figs["norm"].add_trace(
#             go.Scatter(
#                 x=durations,
#                 y=distorted_pulse,
#                 name=f"Distorted A = {amp:.3f}",
#                 mode="lines+markers",
#                 marker=dict(size=5),
#             ),
#             row=1,
#             col=1,
#         )

#         # fit over amplitudes vs time
#         ideal_pulse = np.ones(len(durations))
#         start_fit = 40 - 4
#         end_fit = 165 - 4  # len(durations)
#         amp_guess = -0.78
#         tau_guess = 37
#         pguess = [amp_guess, tau_guess]
#         b_fit, a_fit, popt = cryoscope_fit_exp(
#             durations=durations[start_fit:end_fit],
#             ideal_pulse=ideal_pulse[start_fit:end_fit],
#             distorted_pulse=distorted_pulse[start_fit:end_fit],
#             pguess=pguess,
#         )  # FIXME
#         # print(b_fit, a_fit)
#         reconstructed_pulse = cryoscope_step_response_exp(
#             t=durations[start_fit:end_fit], p=popt, pulse=ideal_pulse[start_fit:end_fit]
#         )

#         # simulated_step_extrap = cryoscope_step_response_exp(t=durations,
#         #                                             p=popt,
#         #                                             awg_reconstructed_pulse=ideal_pulse)

#         # initial_guess = cryoscope_step_response_exp(t=durations,
#         #                                             p=[0.78,37],
#         #                                             awg_reconstructed_pulse=ideal_pulse)

#         figs["norm"].add_trace(
#             go.Scatter(
#                 x=durations[start_fit:end_fit],
#                 y=ideal_pulse,
#                 name=f"Ideal A = {amp:.3f}",
#                 mode="lines+markers",
#                 marker=dict(size=5),  # , color="lightcoral"),
#             ),
#             row=1,
#             col=1,
#         )

#         figs["norm"].add_trace(
#             go.Scatter(
#                 x=durations[start_fit:end_fit],
#                 y=reconstructed_pulse,
#                 name=f"Reconstructed A = {amp:.3f}",
#                 mode="lines+markers",
#                 marker=dict(size=5),  # , color="lightcoral"),
#             ),
#             row=1,
#             col=1,
#         )

#     figs["norm"].update_layout(
#         xaxis_title=title_x,
#         yaxis_title=title_y,
#         title=f"{global_title}",
#     )
#     fitting_report = "No fitting data"
#     return [figs["norm"]], fitting_report


# # Helper functions


# def smooth(x, window_len=11, window="hanning"):
#     """smooth the data using a window with requested size.
#     This method is based on the convolution of a scaled window with the
#         signal.
#     The signal is prepared by introducing reflected copies of the signal
#     (with the window size) in both ends so that transient parts are
#         minimized
#     in the begining and end part of the output signal.
#     input:
#         x: the input signal
#         window_len: the dimension of the smoothing window; should be an
#             odd integer
#         window: the type of window from 'flat', 'hanning', 'hamming',
#             'bartlett', 'blackman'
#         flat window will produce a moving average smoothing.
#     output:
#         the smoothed signal
#     example:
#     t=linspace(-2,2,0.1)
#     x=sin(t)+randn(len(t))*0.1
#     y=smooth(x)
#     see also:
#     numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman,
#     numpy.convolve
#     scipy.signal.lfilter
#     """
#     if int(window_len) & 0x1 == 0:
#         window_len += 1

#     if x.ndim != 1:
#         raise ValueError("smooth only accepts 1 dimension arrays.")

#     if x.size < window_len:
#         raise ValueError("Input vector needs to be bigger than window size.")

#     if window_len < 3:
#         return x

#     if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
#         raise ValueError(
#             "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
#         )

#     s = np.r_[x[window_len - 1 : 0 : -1], x, x[-1:-window_len:-1]]

#     if window == "flat":  # moving average
#         w = np.ones(window_len, "d")

#     else:
#         w = eval("np." + window + "(window_len)")
#     y = np.convolve(w / w.sum(), s, mode="valid")

#     # Cut edges of y since a mirror image is used
#     edge = (window_len - 1) / 2
#     edge = int(edge)
#     return y[edge:-edge]


# def coefs_from_exp(amp, tau):
#     alpha = 1 - np.exp(-1 / (tau * (1 + amp)))
#     # the coefficient k (and the filter transfer function)
#     # depend on the sign of amp

#     if amp >= 0.0:
#         k = amp / (1 + amp - alpha)
#         # compensating overshoot by a low-pass filter
#         # correction filter y[n] = (1-k)*x[n] + k*u[n] = x[n] + k*(u[n]-x[n])
#         # where u[n] = u[n-1] + alpha*(x[n] - u[n-1])
#     else:
#         k = amp / (1 + amp) / (1 - alpha)
#         # compensating low-pass by an overshoot
#         # correction filter y[n] = (1+k)*x[n] - k*u[n] = x[n] - k*(u[n]-x[n])
#         # where u[n] = u[n-1] + alpha*(x[n] - u[n-1])

#     b = [(1 - k + k * alpha), -(1 - k) * (1 - alpha)]

#     # while the denominator stays the same
#     a = [1, -(1 - alpha)]
#     return b, a


# def cryoscope_step_response_exp(t, p, pulse):
#     # This function returns the convolution of a filer h(t,params=p)
#     #   applied to meassured distorted pulse -> should return rectangular
#     #   applied to a rectangular pulse -> returns a predistorted pulse
#     # It is done via lfilter function, and using an exponential model for h(t,p)

#     # https://arxiv.org/pdf/1907.04818.pdf (page 11 - filter formula S22)
#     # p = [A, tau_iir]
#     # p = [b0 = 1−k +k ·α, b1 = −(1−k)·(1−α),a0 = 1 and a1 = −(1−α)]
#     # p = [b0, b1, a0, a1]
#     from scipy.signal import lfilter

#     amp = p[0]  # ration between the shooting point and the settling amplitude
#     tau_iir = p[1]
#     b_fit, a_fit = coefs_from_exp(amp, tau_iir)
#     return lfilter(b_fit, a_fit, pulse)


# def cryoscope_fit_exp(durations, ideal_pulse, distorted_pulse, pguess):
#     # print('calling curve_fit')
#     # print(pguess)
#     amp_guess = pguess[0]
#     tau_guess = pguess[1]

#     def wrapper(t, amp, tau, gain):
#         return gain * cryoscope_step_response_exp(t, [amp, tau], distorted_pulse)

#     fit_mod = Model(wrapper)
#     fit_mod.set_param_hint("amp", value=amp_guess, min=-2, max=2, vary=True)
#     fit_mod.set_param_hint("tau", value=tau_guess, min=0, max=1000, vary=True)
#     fit_mod.set_param_hint("gain", value=1, min=0, max=2, vary=True)
#     params = fit_mod.make_params()
#     fit_result = fit_mod.fit(data=ideal_pulse, t=durations, params=params)
#     print(fit_result.fit_report())
#     # popt, pcov = curve_fit(, x_data, y_data, p0=pguess)
#     print(fit_result.best_values)
#     fit_result.plot_fit()
#     amp = fit_result.best_values[
#         "amp"
#     ]  # ration between the shooting point and the settling amplitude
#     tau_iir = fit_result.best_values["tau"]
#     b_fit, a_fit = coefs_from_exp(amp, tau_iir)
#     return b_fit, a_fit, [amp, tau_iir]

#     # def cryoscope_fit(x_data, y_data, awg_reconstructed_pulse):
#     #     pguess = [
#     #         1, #b0
#     #         0, #b1
#     #         1, #a0
#     #         0  #a1
#     #     ]
#     #     popt, pcov = curve_fit(lambda t,*p:cryoscope_step_response(t, p, awg_reconstructed_pulse), x_data, y_data, p0=pguess)
#     #     b_fit = popt[[0,1]]
#     #     a_fit = popt[[2,3]]
#     #     return b_fit, a_fit

#     # def cryoscope_step_response(t, p, awg_reconstructed_pulse):

#     #     # https://arxiv.org/pdf/1907.04818.pdf (page 11 - filter formula S22)
#     #     # p = [b0 = 1−k +k ·α, b1 = −(1−k)·(1−α),a0 = 1 and a1 = −(1−α)]
#     #     # p = [b0, b1, a0, a1]
#     #     from scipy.signal import lfilter
#     #     b_fit = p[:2]
#     #     a_fit = p[2:]
#     #     return lfilter(b_fit, a_fit, awg_reconstructed_pulse)

#     # def cryoscope_step_response_exp(t, p, awg_reconstructed_pulse):

#     #     # https://arxiv.org/pdf/1907.04818.pdf (page 11 - filter formula S22)
#     #     # p = [A, tau_iir]
#     #     # p = [b0 = 1−k +k ·α, b1 = −(1−k)·(1−α),a0 = 1 and a1 = −(1−α)]
#     #     # p = [b0, b1, a0, a1]
#     #     from scipy.signal import lfilter
#     #     amp = p[0] # ration between the shooting point and the settling amplitude
#     #     tau_iir = p[1]
#     #     alpha = 1-np.exp((1+amp)/tau_iir)
#     #     if amp<0:
#     #         k = amp/((1+amp)*(1-alpha))
#     #     else:
#     #         k = 1/(1+amp-alpha)
#     #     b_fit = [1-k+k*alpha, -(1-k)*(1-alpha)]
#     #     a_fit = [1,-(1-alpha)]
#     #     return lfilter(b_fit, a_fit, awg_reconstructed_pulse)

#     # def cryoscope_fit_exp(x_data, y_data, awg_reconstructed_pulse):
#     #     amp_guess = -0.78
#     #     tau_guess = 37
#     #     pguess = [amp_guess, tau_guess]
#     #     popt, pcov = curve_fit(lambda t,*p:cryoscope_step_response_exp(t, p, awg_reconstructed_pulse), x_data, y_data, p0=pguess)
#     #     amp = popt[0] # ration between the shooting point and the settling amplitude
#     #     tau_iir = popt[1]
#     #     alpha = 1-np.exp((1+amp)/tau_iir)
#     #     if amp<0:
#     #         k = amp/((1+amp)*(1-alpha))
#     #     else:
#     #         k = 1/(1+amp-alpha)
#     #     b_fit = [1-k+k*alpha, -(1-k)*(1-alpha)]
#     #     a_fit = [1,-(1-alpha)]
#     #     return b_fit, a_fit, popt

#     # ?
#     # def cryoscope_detuning_amplitude(folder, routine, qubit, format):
#     #     data = load_data(folder, "data", routine, format, "data")
#     #     import numpy as np
#     #     import scipy.signal as ss

#     #     MX_tag = "MX"
#     #     MY_tag = "MY"

#     #     amplitude = data.get_values("flux_pulse_amplitude", "dimensionless")
#     #     duration = data.get_values("flux_pulse_duration", "ns")
#     #     flux_pulse_duration = duration[data.df["component"] == MY_tag].to_numpy()
#     #     flux_pulse_amplitude = amplitude[data.df["component"] == MY_tag].to_numpy()
#     #     amplitude_unique = flux_pulse_amplitude[
#     #         flux_pulse_duration == flux_pulse_duration[0]
#     #     ]
#     #     duration_unique = flux_pulse_duration[
#     #         flux_pulse_amplitude == flux_pulse_amplitude[0]
#     #     ]

#     #     # Making figure
#     #     figs = {}
#     #     figs["norm"] = make_subplots(
#     #         rows=1,
#     #         cols=1,
#     #         horizontal_spacing=0.1,
#     #         vertical_spacing=0.2,
#     #     )

#     #     mean_detuning = []

#     #     for amp in amplitude_unique:
#     #         x_data = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MX_tag].to_numpy()
#     #         y_data = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MY_tag].to_numpy()

#     #         complex_data = x_data + 1j * y_data
#     #         #print (len(complex_data))
#     #         ws = 10 # Needs to be adjusted for data analysis
#     #         norm_data = normalize_sincos(complex_data, window_size=ws)

#     #         sampling_rate = 1 / (duration_unique[1] - duration_unique[0])
#     #         derivative_window_length = 7 / sampling_rate
#     #         derivative_window_size = max(3, int(derivative_window_length * sampling_rate))
#     #         derivative_window_size += (derivative_window_size + 1) % 2
#     #         derivative_order = 2
#     #         nyquist_order=0

#     #         freq_guess, phase_guess, offset_guess, amp_guess = fft_based_freq_guess_complex(norm_data)
#     #         demod_freq = - \
#     #                 freq_guess * sampling_rate

#     #         #y = amp*exp(2pi i f t + ph) + off.
#     #         #demod_data = amp_guess * np.exp(2 * np.pi * 1j * demod_freq * duration_unique + phase_guess) + offset_guess
#     #         demod_data = np.exp(2 * np.pi * 1j * duration_unique * demod_freq) * norm_data
#     #         # Di Carlo has smooth here!!!

#     #         # Unwrap phase
#     #         phase = np.arctan2(demod_data.imag, demod_data.real)
#     #         # phase_unwrapped = np.unwrap(phase)
#     #         phase_unwrapped = np.unwrap(np.angle(demod_data))

#     #         # use a savitzky golay filter: it take sliding window of length
#     #         # `window_length`, fits a polynomial, returns derivative at
#     #         # middle point
#     #         # phase = phase_unwrapped_data

#     #         # Di Carlo method
#     #         detuning = savgol_filter(
#     #             phase_unwrapped / (2 * np.pi),
#     #             window_length = derivative_window_size,
#     #             polyorder = derivative_order,
#     #             deriv=1) * sampling_rate * 1e9

#     #         nyquist_order = 0
#     #         detuning = get_real_detuning(detuning, demod_freq, sampling_rate, nyquist_order)

#     #         # Maxime method
#     #         # dt = np.diff(duration_unique) * 1e-9
#     #         # dphi_dt_unwrap = np.abs(np.diff(phase_unwrapped) / dt)
#     #         # detuning = dphi_dt_unwrap / (2 * np.pi)

#     #         mean_detuning.append(np.mean(detuning))

#     #     # Mean detunning vs. amplitude
#     #     global_title = "Mean Detuning vs. Amplitude"
#     #     title_x = "Amplitude (dimensionless)"
#     #     title_y = "Detunning mean (Hz)"
#     #     figs["norm"].add_trace(
#     #         go.Scatter(
#     #             x=amplitude_unique,
#     #             y=mean_detuning,
#     #             name=f"Detuning"
#     #         ),
#     #         row=1,
#     #         col=1,
#     #     )

#     #     figs["norm"].update_layout(
#     #         xaxis_title=title_x,
#     #         yaxis_title=title_y,
#     #         title=f"{global_title}",
#     #     )
#     #     fitting_report = "No fitting data"
#     # return [figs["norm"]], fitting_report

#     # def peak_finder_v2(x, y, perc=90, window_len=11):
#     #     """
#     #     Peak finder based on argrelextrema function from scipy
#     #     only finds maximums, this can be changed to minimum by using -y instead of y
#     #     """
#     #     smoothed_y = smooth(y, window_len=window_len)
#     #     percval = np.percentile(smoothed_y, perc) # finds the top perc points in amplitude smoothed_y
#     #     filtered_y = np.where(smoothed_y > percval, smoothed_y, percval)
#     #     array_peaks = argrelextrema(filtered_y, np.greater)
#     #     peaks_x = x[array_peaks]
#     #     sort_mask = np.argsort(y[array_peaks])[::-1]
#     #     return peaks_x[sort_mask]

#     # def normalize_sincos(
#     #         data,
#     #         window_size_frac=500,
#     #         window_size=None,
#     #         do_envelope=True):

#     #     if window_size is None:
#     #         window_size = len(data) // window_size_frac

#     #         # window size for savgol filter must be odd
#     #         window_size -= (window_size + 1) % 2

#     #     mean_data_r = savgol_filter(data.real, window_size, 0, 0)
#     #     mean_data_i = savgol_filter(data.imag, window_size, 0, 0)

#     #     mean_data = mean_data_r + 1j * mean_data_i

#     #     if do_envelope:
#     #         envelope = np.sqrt(
#     #             savgol_filter(
#     #                 (np.abs(
#     #                     data -
#     #                     mean_data))**2,
#     #                 window_size,
#     #                 0,
#     #                 0))
#     #     else:
#     #         envelope = 1
#     #     norm_data = ((data - mean_data) / envelope)
#     #     return norm_data

#     # def fft_based_freq_guess_complex(y):
#     #     """
#     #     guess the shape of a sinusoidal complex signal y (in multiples of
#     #         sampling rate), by selecting the peak in the fft.
#     #     return guess (f, ph, off, amp) for the model
#     #         y = amp*exp(2pi i f t + ph) + off.
#     #     """
#     #     fft = np.fft.fft(y)[1:len(y)]
#     #     freq_guess_idx = np.argmax(np.abs(fft))
#     #     if freq_guess_idx >= len(y) // 2:
#     #         freq_guess_idx -= len(y)

#     #     freq_guess = 1 / len(y) * (freq_guess_idx + 1)
#     #     phase_guess = np.angle(fft[freq_guess_idx]) + np.pi / 2
#     #     amp_guess = np.absolute(fft[freq_guess_idx]) / len(y)
#     #     offset_guess = np.mean(y)

#     #     return freq_guess, phase_guess, offset_guess, amp_guess

#     # def get_real_detuning(detuning, demod_freq, sampling_rate, nyquist_order=0):
#     #     real_detuning = detuning - demod_freq + sampling_rate * nyquist_order
#     #     return real_detuning

#     # def normalize_data(x, y, amplitude_unique, duration_unique, flux_pulse_amplitude, flux_pulse_duration, amplitude):
#     #     x_norm = np.ones((len(amplitude_unique), len(duration_unique))) * np.nan
#     #     y_norm = np.ones((len(amplitude_unique), len(duration_unique))) * np.nan

#     #     for i, amp in enumerate(amplitude_unique):
#     #         idx = np.where(flux_pulse_amplitude == amp)
#     #         if amp == flux_pulse_amplitude[-1]:
#     #             n = int(np.where(flux_pulse_duration[-1] == duration_unique)[0][0]) + 1
#     #         else:
#     #             n = len(duration_unique)

#     #         x_norm[i, :n] = x[idx]
#     #         x_norm[i, :n] = x_norm[i, :n] - 0.5
#     #         x_norm[i, :n] = x_norm[i, :n] / max(abs(x_norm[i, :n]))
#     #         y_norm[i, :n] = y[idx]
#     #         y_norm[i, :n] = y_norm[i, :n] - 0.5
#     #         y_norm[i, :n] = y_norm[i, :n] / max(abs(y_norm[i, :n]))

#     #     return x_norm, y_norm

#     # def normalize_sincos(x, y, window_size_frac=61, window_size=None, do_envelope=True):

#     #     if window_size is None:
#     #         window_size = len(x) // window_size_frac

#     #         # window size for savgol filter must be odd
#     #         window_size -= (window_size + 1) % 2

#     #     x = savgol_filter(x, window_size, 0, 0)  ## why poly order 0
#     #     y = savgol_filter(y, window_size, 0, 0)

#     #     # if do_envelope:
#     #     #     envelope = np.sqrt(savgol_filter((np.abs(data - mean_data))**2, window_size, 0, 0))
#     #     # else:
#     #     #     envelope = 1

#     #     # norm_data = ((data - mean_data) / envelope)

#     #     return x, y

#     # def cryoscope_demod_circle(folder, routine, qubit, format):
#     #     data = load_data(folder, "data", routine, format, "data")
#     #     import numpy as np
#     #     import scipy.signal as ss

#     #     MX_tag = "MX"
#     #     MY_tag = "MY"

#     #     amplitude = data.get_values("flux_pulse_amplitude", "dimensionless")
#     #     duration = data.get_values("flux_pulse_duration", "ns")
#     #     flux_pulse_duration = duration[data.df["component"] == MY_tag].to_numpy()
#     #     flux_pulse_amplitude = amplitude[data.df["component"] == MY_tag].to_numpy()
#     #     amplitude_unique = flux_pulse_amplitude[
#     #         flux_pulse_duration == flux_pulse_duration[0]
#     #     ]
#     #     duration_unique = flux_pulse_duration[
#     #         flux_pulse_amplitude == flux_pulse_amplitude[0]
#     #     ]

#     #     # Making figure
#     #     figs = {}
#     #     figs["norm"] = make_subplots(
#     #         rows=1,
#     #         cols=1,
#     #         horizontal_spacing=0.1,
#     #         vertical_spacing=0.2,
#     #     )

#     #     for amp in amplitude_unique:
#     #         x_data = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MX_tag].to_numpy()
#     #         y_data = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MY_tag].to_numpy()

#     #         complex_data = x_data + 1j * y_data
#     #         #print (len(complex_data))
#     #         ws = 10 # Needs to be adjusted for data analysis
#     #         norm_data = normalize_sincos(complex_data, window_size=ws)

#     #         sampling_rate = 1 / (duration_unique[1] - duration_unique[0])
#     #         derivative_window_length = 7 / sampling_rate
#     #         derivative_window_size = max(3, int(derivative_window_length * sampling_rate))
#     #         derivative_window_size += (derivative_window_size + 1) % 2
#     #         derivative_order = 2
#     #         nyquist_order=0

#     #         freq_guess, phase_guess, offset_guess, amp_guess = fft_based_freq_guess_complex(norm_data)
#     #         demod_freq = - \
#     #                 freq_guess * sampling_rate

#     #         #y = amp*exp(2pi i f t + ph) + off.
#     #         #demod_data = amp_guess * np.exp(2 * np.pi * 1j * demod_freq * duration_unique + phase_guess) + offset_guess
#     #         demod_data = np.exp(2 * np.pi * 1j * duration_unique * demod_freq) * norm_data
#     #         # Di Carlo has smooth here!!!

#     #         # demod_data circle
#     #         global_title = "FFT demod data circle"
#     #         title_x = "<X>"
#     #         title_y = "<Y>"
#     #         figs["norm"].add_trace(
#     #             go.Scatter(
#     #                 x=demod_data.real,
#     #                 y=demod_data.imag,
#     #                 name=f"A = {amp:.3f}"
#     #             ),
#     #             row=1,
#     #             col=1,
#     #         )

#     #     figs["norm"].update_layout(
#     #         xaxis_title=title_x,
#     #         yaxis_title=title_y,
#     #         title=f"{global_title}",
#     #     )
#     #     fitting_report = "No fitting data"
#     # return [figs["norm"]], fitting_report

#     # def cryoscope_demod_fft(folder, routine, qubit, format):
#     #     data = load_data(folder, "data", routine, format, "data")
#     #     import numpy as np
#     #     import scipy.signal as ss

#     #     MX_tag = "MX"
#     #     MY_tag = "MY"

#     #     amplitude = data.get_values("flux_pulse_amplitude", "dimensionless")
#     #     duration = data.get_values("flux_pulse_duration", "ns")
#     #     flux_pulse_duration = duration[data.df["component"] == MY_tag].to_numpy()
#     #     flux_pulse_amplitude = amplitude[data.df["component"] == MY_tag].to_numpy()
#     #     amplitude_unique = flux_pulse_amplitude[
#     #         flux_pulse_duration == flux_pulse_duration[0]
#     #     ]
#     #     duration_unique = flux_pulse_duration[
#     #         flux_pulse_amplitude == flux_pulse_amplitude[0]
#     #     ]

#     #     # Making figure
#     #     figs = {}
#     #     figs["norm"] = make_subplots(
#     #         rows=1,
#     #         cols=1,
#     #         horizontal_spacing=0.1,
#     #         vertical_spacing=0.2,
#     #     )

#     #     for amp in amplitude_unique:
#     #         x_data = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MX_tag].to_numpy()
#     #         y_data = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MY_tag].to_numpy()

#     #         complex_data = x_data + 1j * y_data
#     #         #print (len(complex_data))
#     #         ws = 10 # Needs to be adjusted for data analysis
#     #         norm_data = normalize_sincos(complex_data, window_size=ws)

#     #         sampling_rate = 1 / (duration_unique[1] - duration_unique[0])
#     #         derivative_window_length = 7 / sampling_rate
#     #         derivative_window_size = max(3, int(derivative_window_length * sampling_rate))
#     #         derivative_window_size += (derivative_window_size + 1) % 2
#     #         derivative_order = 2
#     #         nyquist_order=0

#     #         freq_guess, phase_guess, offset_guess, amp_guess = fft_based_freq_guess_complex(norm_data)
#     #         demod_freq = - \
#     #                 freq_guess * sampling_rate

#     #         #y = amp*exp(2pi i f t + ph) + off.
#     #         #demod_data = amp_guess * np.exp(2 * np.pi * 1j * demod_freq * duration_unique + phase_guess) + offset_guess
#     #         demod_data = np.exp(2 * np.pi * 1j * duration_unique * demod_freq) * norm_data
#     #         # Di Carlo has smooth here!!!

#     #         # <X> sin <Y> cos demod data
#     #         global_title = "<X> , <Y> after FFT demodulation"
#     #         title_x = "Flux Pulse duration"
#     #         title_y = "<X> , <Y>"
#     #         figs["norm"].add_trace(
#     #             go.Scatter(
#     #                 x=duration_unique,
#     #                 y=demod_data.real,
#     #                 name=f"<X> fft demod | A = {amp:.3f}"
#     #             ),
#     #             row=1,
#     #             col=1,
#     #         )

#     #         figs["norm"].add_trace(
#     #             go.Scatter(
#     #                 x=duration_unique,
#     #                 y=demod_data.imag,
#     #                 name=f"<Y> fft demod | A = {amp:.3f}"
#     #             ),
#     #             row=1,
#     #             col=1,
#     #         )

#     #     figs["norm"].update_layout(
#     #         xaxis_title=title_x,
#     #         yaxis_title=title_y,
#     #         title=f"{global_title}",
#     #     )
#     #     fitting_report = "No fitting data"
#     # return [figs["norm"]], fitting_report
