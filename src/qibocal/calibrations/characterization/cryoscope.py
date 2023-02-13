from typing import Optional

import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import FluxPulse, PulseSequence, Rectangular
from qibolab.sweeper import Sweeper

from qibocal import plots
from qibocal.data import Data, DataUnits
from qibocal.decorators import plot


@plot("Flux pulse timing", plots.flux_pulse_timing)
def flux_pulse_timing(
    platform: AbstractPlatform,
    qubits: list,
    flux_pulse_amplitude_start,
    flux_pulse_amplitude_end,
    flux_pulse_amplitude_step,
    flux_pulse_start_start,
    flux_pulse_start_end,
    flux_pulse_start_step,
    flux_pulse_duration,
    time_window,
    nshots=1024,
    relaxation_time=None,
    software_averages=1,
    points=10,
):
    r"""
    The flux pulse timing experiment aims to determine the relative time of flight between
    the drive lines and the flux lines, as well as the time required for the transient response to
    disappear when the flux pulse stops. This is later required for the correct scheduling of
    the flux pulses in the implementation of 2q gates.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (list): List of target qubits to perform the action
        flux_pulse_amplitude_start (int): Initial flux pulse amplitude
        flux_pulse_amplitude_end (int): Maximum flux pulse amplitude
        flux_pulse_amplitude_step (int): Scan range step for the flux pulse amplitude
        flux_pulse_start_start (int): The earliest start time of the flux pluse, in ns,
            relative to the start of the first pulse of the sequence Ry(pi/2)
        flux_pulse_start_end (int): The earliest start time of the flux pluse, in ns
        flux_pulse_start_step (int): Scan range step for the flux pulse start
        flux_pulse_duration (int): The duration of the flux pulse
        time_window (int): The time window in ns between the end of the first pulse of the sequence Ry(pi/2)
            and the beginning of the second pulse of the sequence Ry(pi/2)
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
            - **flux_pulse_start[ns]**:The absolut start time of the flux pulse, in ns
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages

        - A Data object with the following parameters
            - **flux_pulse_duration**: The duration of the flux pulse
            - **time_window**: The time window in ns between the end of the first pulse of the sequence Ry(pi/2)
                    and the beginning of the second pulse of the sequence Ry(pi/2)
            - **initial_RY90_pulses_finish**: The absolut finish time of the first pulse of the sequence Ry(pi/2)
            - **qubit**: The qubit being tested

    """

    # the purpose of this routine is to determine the time of flight of a flux pulse and the duration
    # of the transient at the end of the pulse

    # 1) from state |0> apply Ry(pi/2) to state |+>,
    # 2) apply a detunning flux pulse of fixed duration at various start times
    # 3) after a certain time window from the end of the initial Ry(pi/2) pulse,
    #    measure in the X axis: Ry(pi/2) - MZ

    #   MX = Ry(pi/2) - (flux) - Ry(pi/2)  - MZ
    # If the flux pulse of sufficient amplitude falls within the time window between the Ry(pi/2) pulses,
    # it detunes the qubit and results in a rotation around the Z axis r so that MX = Cos(r)

    # reload instrument settings from runcard
    platform.reload_settings()

    # define the sequences of pulses to be executed
    sequence = PulseSequence()

    initial_RY90_pulses = {}
    flux_pulses = {}
    RY90_pulses = {}
    MZ_ro_pulses = {}
    if flux_pulse_start_start < 0:
        initial_RY90_pulses_start = -flux_pulse_start_start
    else:
        initial_RY90_pulses_start = 0

    for qubit in qubits:
        # start at |+> by rotating Ry(pi/2)
        initial_RY90_pulses[qubit] = platform.create_RX90_pulse(
            qubit, start=initial_RY90_pulses_start, relative_phase=np.pi / 2
        )

        # wait time window

        # rotate around the Y asis Ry(pi/2) to meassure X component
        RY90_pulses[qubit] = platform.create_RX90_pulse(
            qubit,
            start=initial_RY90_pulses[qubit].finish + time_window,
            relative_phase=np.pi / 2,
        )

        # add ro pulse at the end of each sequence
        MZ_ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RY90_pulses[qubit].finish
        )

        # apply a detuning flux pulse around the time window
        flux_pulses[qubit] = FluxPulse(
            start=initial_RY90_pulses_start + flux_pulse_start_start,
            duration=flux_pulse_duration,
            amplitude=flux_pulse_amplitude_start,  # fix for each run
            shape=Rectangular(),
            # relative_phase=0,
            channel=qubit.flux.name,
            qubit=qubit.name,
        )

        # add pulses to the sequences
        sequence.add(
            initial_RY90_pulses[qubit],
            flux_pulses[qubit],
            RY90_pulses[qubit],
            MZ_ro_pulses[qubit],
        )

    # define the parameters to sweep and their range:
    # flux pulse amplitude
    # flux pulse duration

    flux_pulse_amplitude_range = np.arange(
        flux_pulse_amplitude_start, flux_pulse_amplitude_end, flux_pulse_amplitude_step
    )
    sweeper = Sweeper(
        "amplitude",
        flux_pulse_amplitude_range,
        pulses=[flux_pulses[qubit] for qubit in qubits],
    )
    flux_pulse_start_range = np.arange(
        flux_pulse_start_start, flux_pulse_start_end, flux_pulse_start_step
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include flux pulse duration, amplitude and the probability
    data = DataUnits(
        name=f"data",
        quantities={
            "flux_pulse_amplitude": "dimensionless",
            "flux_pulse_start": "ns",
            "prob": "dimensionless",
        },
        options=["qubit", "iteration"],
    )

    parameters = Data(
        name=f"parameters",
        quantities=[
            "flux_pulse_duration",
            "time_window",
            "initial_RY90_pulses_finish",
            "qubit",
        ],
    )
    for qubit in qubits:
        parameters.add(
            {
                "flux_pulse_duration": flux_pulse_duration,
                "time_window": time_window,
                "initial_RY90_pulses_finish": initial_RY90_pulses[qubit].finish,
                "qubit": qubit,
            }
        )
    yield parameters

    ndata = len(flux_pulse_amplitude_range)
    count = 0
    for iteration in range(software_averages):
        # sweep the parameters
        for start in flux_pulse_start_range:
            # save data as often as defined by points
            if count % points == 0 and count > 0:
                # save data
                yield data
            for qubit in qubits:
                flux_pulses[qubit].start = initial_RY90_pulses_start + start
            # execute the pulse sequence
            results = platform.sweep(
                sequence,
                sweeper,
                nshots=nshots,
                relaxation_time=relaxation_time,
                average=False,
            )
            for qubit in qubits:
                qubit_res = results[MZ_ro_pulses[qubit].serial]
                r = {
                    "MSR[V]": qubit_res.measurement.mean(axis=0),
                    "i[V]": qubit_res.i.mean(axis=0),
                    "q[V]": qubit_res.q.mean(axis=0),
                    "phase[rad]": qubit_res.phase.mean(axis=0),
                    "prob[dimensionless]": qubit_res.shots.mean(axis=0),
                    "flux_pulse_amplitude[dimensionless]": flux_pulse_amplitude_range,
                    "flux_pulse_start[ns]": ndata * [start],
                    "qubit": ndata * [qubit],
                    "iteration": ndata * [iteration],
                }
                data.add_data_from_dict(r)

            count += 1
    # finally, save the remaining data
    yield data


@plot("cryoscope_raw", plots.cryoscope_raw)
@plot("cryoscope_dephasing_heatmap", plots.cryoscope_dephasing_heatmap)
@plot("cryoscope_fft_peak_fitting", plots.cryoscope_fft_peak_fitting)
@plot("cryoscope_fft", plots.cryoscope_fft)
@plot("cryoscope_phase", plots.cryoscope_phase)
@plot("cryoscope_phase_heatmap", plots.cryoscope_phase_heatmap)
@plot("cryoscope_phase_unwrapped", plots.cryoscope_phase_unwrapped)
@plot("cryoscope_phase_unwrapped_heatmap", plots.cryoscope_phase_unwrapped_heatmap)
@plot(
    "cryoscope_phase_amplitude_unwrapped_heatmap",
    plots.cryoscope_phase_amplitude_unwrapped_heatmap,
)
@plot("cryoscope_detuning_time", plots.cryoscope_detuning_time)
@plot("cryoscope_distorted_amplitude_time", plots.cryoscope_distorted_amplitude_time)
@plot(
    "cryoscope_reconstructed_amplitude_time",
    plots.cryoscope_reconstructed_amplitude_time,
)
def cryoscope(
    platform: AbstractPlatform,
    qubits: dict,
    flux_pulse_amplitude,
    flux_pulse_duration_start,
    flux_pulse_duration_end,
    flux_pulse_duration_step,
    flux_pulse_amplitude_start,
    flux_pulse_amplitude_end,
    flux_pulse_amplitude_step,
    delay_before_readout,
    flux_pulse_shapes: Optional[list] = None,
    nshots=1024,
    relaxation_time=None,
    software_averages=1,
    points=10,
):
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
        delay_before_readout (int): A time delay in ns between the end of the flux pulse and the beginning of the measurement of the phase
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
    #   MX = Ry(pi/2) - (flux)(t) - wait - Ry(pi/2)  - MZ
    #   MY = Ry(pi/2) - (flux)(t) - wait - Rx(-pi/2) - MZ
    # The flux pulse detunes the qubit and results in a rotation around the Z axis = atan(MY/MX)

    # reload instrument settings from runcard
    platform.reload_settings()

    # define the sequences of pulses to be executed
    MX_seq = PulseSequence()
    MY_seq = PulseSequence()

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

        # apply a detuning flux pulse
        flux_pulses[qubit] = FluxPulse(
            start=initial_RY90_pulses[qubit].se_finish,
            duration=flux_pulse_duration_start,  # sweep to produce oscillations [up to 400ns] in steps od 1ns? or 4?
            amplitude=flux_pulse_amplitude,  # fix for each run
            shape=flux_pulse_shape,
            # relative_phase=0,
            channel=platform.qubits[qubit].flux.name,
            qubit=platform.qubits[qubit].name,
        )

        # wait delay_before_readout

        # rotate around the X asis Rx(-pi/2) to meassure Y component
        RX90_pulses[qubit] = platform.create_RX90_pulse(
            qubit,
            start=flux_pulses[qubit].finish + delay_before_readout,
            relative_phase=np.pi,
        )

        # rotate around the Y asis Ry(pi/2) to meassure X component
        RY90_pulses[qubit] = platform.create_RX90_pulse(
            qubit,
            start=flux_pulses[qubit].finish + delay_before_readout,
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

        # DEBUG: Plot Cryoscope Sequences
        # MX_seq.plot("MX_seq")
        # MY_seq.plot("MY_seq")

    MX_tag = "MX"
    MY_tag = "MY"

    # define the parameters to sweep and their range:
    # flux pulse amplitude
    # flux pulse duration

    flux_pulse_amplitude_range = np.arange(
        flux_pulse_amplitude_start, flux_pulse_amplitude_end, flux_pulse_amplitude_step
    )
    sweeper = Sweeper(
        "amplitude",
        flux_pulse_amplitude_range,
        pulses=[flux_pulses[qubit] for qubit in qubits],
    )
    flux_pulse_duration_range = np.arange(
        flux_pulse_duration_start, flux_pulse_duration_end, flux_pulse_duration_step
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include flux pulse duration, amplitude and the probability
    data = DataUnits(
        name=f"data",
        quantities={
            "flux_pulse_duration": "ns",
            "flux_pulse_amplitude": "dimensionless",
            "prob": "dimensionless",
        },
        options=["component", "qubit", "iteration"],
    )

    ndata = len(flux_pulse_amplitude_range)
    count = 0
    for iteration in range(software_averages):
        for duration in flux_pulse_duration_range:
            # save data as often as defined by points
            if count % points == 0 and count > 0:
                # save data
                yield data
            for qubit in qubits:
                flux_pulses[qubit].duration = duration
            # execute the pulse sequences
            for sequence, tag in [(MX_seq, MX_tag), (MY_seq, MY_tag)]:
                results = platform.sweep(
                    sequence,
                    sweeper,
                    nshots=nshots,
                    relaxation_time=relaxation_time,
                    average=False,
                )
                for qubit in qubits:
                    qubit_res = results[MZ_ro_pulses[qubit].serial]

                    r = {
                        "MSR[V]": qubit_res.measurement.mean(axis=0),
                        "i[V]": qubit_res.i.mean(axis=0),
                        "q[V]": qubit_res.q.mean(axis=0),
                        "phase[rad]": qubit_res.phase.mean(axis=0),
                        "prob[dimensionless]": qubit_res.shots.mean(axis=0),
                        "flux_pulse_duration[ns]": ndata * [duration],
                        "flux_pulse_amplitude[dimensionless]": flux_pulse_amplitude_range,
                        "component": ndata * [tag],
                        "qubit": ndata * [qubit],
                        "iteration": ndata * [iteration],
                    }
                    data.add_data_from_dict(r)
            count += 1

    # finally, save the remaining data
    yield data
