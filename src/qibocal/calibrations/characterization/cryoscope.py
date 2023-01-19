import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import IIR, FluxPulse, Pulse, PulseSequence, PulseType, Rectangular

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot


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
    qubits: list,
    flux_pulse_duration_start,
    flux_pulse_duration_end,
    flux_pulse_duration_step,
    flux_pulse_amplitude_start,
    flux_pulse_amplitude_end,
    flux_pulse_amplitude_step,
    delay_before_readout,
    flux_pulse_shapes: list = None,
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
        - A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

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
            flux_pulse_shapes[qubit] = eval(flux_pulse_shapes[qubit])
        else:
            flux_pulse_shapes[qubit] = Rectangular()

        # apply a detuning flux pulse
        flux_pulses[qubit] = FluxPulse(
            start=initial_RY90_pulses[qubit].se_finish,
            duration=flux_pulse_duration_start,  # sweep to produce oscillations [up to 400ns] in steps od 1ns? or 4?
            amplitude=flux_pulse_amplitude_start,  # fix for each run
            shape=flux_pulse_shapes[qubit],
            # relative_phase=0,
            channel=platform.qubit_channel_map[qubit][2],
            qubit=qubit,
        )

        # wait delay_before_readout

        # rotate around the X asis Rx(-pi/2) to meassure Y component
        RX90_pulses[qubit] = platform.create_RX90_pulse(
            qubit,
            start=initial_RY90_pulses[qubit].duration
            + flux_pulse_duration_end
            + delay_before_readout,
            relative_phase=np.pi,
        )

        # rotate around the Y asis Ry(pi/2) to meassure X component
        RY90_pulses[qubit] = platform.create_RX90_pulse(
            qubit,
            start=initial_RY90_pulses[qubit].duration
            + flux_pulse_duration_end
            + delay_before_readout,
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
        MX_seq.plot("MX_seq")
        MY_seq.plot("MY_seq")

    MX_tag = "MX"
    MY_tag = "MY"

    # define the parameters to sweep and their range:
    # flux pulse amplitude
    # flux pulse duration

    flux_pulse_amplitude_range = np.arange(
        flux_pulse_amplitude_start, flux_pulse_amplitude_end, flux_pulse_amplitude_step
    )
    flux_pulse_duration_range = np.arange(
        flux_pulse_duration_start, flux_pulse_duration_end, flux_pulse_duration_step
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include wait time and t_max
    data = DataUnits(
        name=f"data",
        quantities={
            "flux_pulse_duration": "ns",
            "flux_pulse_amplitude": "dimensionless",
            "prob": "dimensionless",
        },
        options=["component", "qubit", "iteration"],
    )

    count = 0
    for iteration in range(software_averages):
        # sweep the parameters
        for amplitude in flux_pulse_amplitude_range:
            for duration in flux_pulse_duration_range:
                # save data as often as defined by points
                if count % points == 0 and count > 0:
                    # save data
                    yield data
                for qubit in qubits:
                    flux_pulses[qubit].amplitude = amplitude
                    flux_pulses[qubit].duration = duration

                # execute the pulse sequences
                for sequence, tag in [(MX_seq, MX_tag), (MY_seq, MY_tag)]:
                    results = platform.execute_pulse_sequence(sequence)

                    for qubit in qubits:
                        prob = results["probability"][MZ_ro_pulses[qubit].serial]
                        voltages = results["demodulated_integrated_averaged"][
                            MZ_ro_pulses[qubit].serial
                        ]
                        r = {
                            "MSR[V]": voltages[0],
                            "phase[rad]": voltages[1],
                            "i[V]": voltages[2],
                            "q[V]": voltages[3],
                            "prob[dimensionless]": prob,
                            "flux_pulse_duration[ns]": duration,
                            "flux_pulse_amplitude[dimensionless]": amplitude,
                            "component": tag,
                            "qubit": qubit,
                            "iteration": iteration,
                        }
                        data.add(r)

                count += 1
    # finally, save the remaining data
    yield data
